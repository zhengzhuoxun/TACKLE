#!/usr/bin/env bash
#
# End-to-end generation script:
# 1. Generate instance.json from kgschema.yaml (with optional spec overrides)
# 2. Load instances.json into Neo4j (marker-scoped; see load_instances_to_neo4j.py)
# 3. Generate question_answers.json by executing Cypher templates against
#    that loaded graph
# 4. Generate table CSVs from instance.json and table_schema.yaml
# 5. Assemble MMQA-format JSON + KG statistics
#
# All outputs are placed in a new folder created alongside the meta folder.
#

set -euo pipefail

# Default values
SEED=42
NUM_PER_TYPE=10
TYPE_COUNTS=""
META_DIR=""
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASS="password"

# Print usage
usage() {
    cat <<EOF
Usage: $0 -m <meta_folder> [--seed <int>] [--num-per-type <int>] [--type-counts <json>]
          [--neo4j-uri <uri>] [--neo4j-user <user>] [--neo4j-pass <pass>]

Required:
  -m, --meta          Path to the folder containing the three YAML files:
                      kgschema.yaml, question_templates.yaml, table_schema.yaml

Optional:
  --seed              Random seed (default: 42)
  --num-per-type      Base number of questions per type (default: 10)
  --type-counts       JSON dict specifying exact count per type, e.g.,
                      '{"1-hop retrieval":15,"Flat Aggregation":8,"Nested Aggregation":12}'
                      Overrides specify.yaml for overlapping keys.
  --neo4j-uri         Neo4j bolt URI (default: bolt://localhost:7687)
  --neo4j-user        Neo4j username (default: neo4j)
  --neo4j-pass        Neo4j password (default: password)
  -h, --help          Show this help message
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--meta)
            META_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num-per-type)
            NUM_PER_TYPE="$2"
            shift 2
            ;;
        --type-counts)
            TYPE_COUNTS="$2"
            shift 2
            ;;
        --neo4j-uri)
            NEO4J_URI="$2"
            shift 2
            ;;
        --neo4j-user)
            NEO4J_USER="$2"
            shift 2
            ;;
        --neo4j-pass)
            NEO4J_PASS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [[ -z "$META_DIR" ]]; then
    echo "Error: Missing -m argument."
    usage
fi

# Verify meta directory exists
if [[ ! -d "$META_DIR" ]]; then
    echo "Error: Meta directory not found: $META_DIR"
    exit 1
fi

# Define expected file names (fixed)
SCHEMA_FILE="$META_DIR/kgschema.yaml"
TEMPLATE_FILE="$META_DIR/question_templates.yaml"
TABLE_SCHEMA_FILE="$META_DIR/table_schema.yaml"
SPEC_FILE="$META_DIR/specify.yaml"

# Verify required schema files exist (except specify.yaml which is optional)
for f in "$SCHEMA_FILE" "$TEMPLATE_FILE" "$TABLE_SCHEMA_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: Required file not found: $f"
        exit 1
    fi
done

# Determine script directory (where this script resides)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python scripts are assumed to be in the same directory as this script
GEN_INSTANCE_PY="$SCRIPT_DIR/generate_instance.py"
LOAD_NEO4J_PY="$SCRIPT_DIR/load_instances_to_neo4j.py"
GEN_QUESTION_ANSWER_PY="$SCRIPT_DIR/generate_question_answer.py"
GEN_TABLES_PY="$SCRIPT_DIR/generate_tables.py"
ASSEMBLE_SELF_PY="$SCRIPT_DIR/assemble_self.py"

for py in "$GEN_INSTANCE_PY" "$LOAD_NEO4J_PY" "$GEN_QUESTION_ANSWER_PY" "$GEN_TABLES_PY" "$ASSEMBLE_SELF_PY"; do
    if [[ ! -f "$py" ]]; then
        echo "Error: Python script not found: $py"
        exit 1
    fi
done

# Determine parent directory of meta folder
PARENT_DIR="$(cd "$(dirname "$META_DIR")" && pwd)"

# Create a new output folder alongside meta, with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$PARENT_DIR/output_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"

# ----------------------------------------------------------------------
# Step 1: Generate instance.json (with optional spec overrides)
# ----------------------------------------------------------------------
echo "Step 1: Generating instances.json from $SCHEMA_FILE ..."
CMD_INSTANCE=("python" "$GEN_INSTANCE_PY" "-i" "$SCHEMA_FILE" "-o" "$OUTPUT_DIR/instances.json" "--seed" "$SEED")
if [[ -f "$SPEC_FILE" ]]; then
    CMD_INSTANCE+=("--spec" "$SPEC_FILE")
    echo "Found $SPEC_FILE, applying entity count overrides."
fi
"${CMD_INSTANCE[@]}"
if [[ $? -ne 0 ]]; then
    echo "Error: Instance generation failed."
    exit 1
fi
echo "Done."

# ----------------------------------------------------------------------
# Step 2: Load instances.json into Neo4j (marker-scoped: only touches
# nodes/relationships this script created, safe on a shared Neo4j instance)
# ----------------------------------------------------------------------
echo "Step 2: Loading instances.json into Neo4j ($NEO4J_URI) ..."
python "$LOAD_NEO4J_PY" \
    -i "$OUTPUT_DIR/instances.json" \
    -s "$SCHEMA_FILE" \
    -t "$TEMPLATE_FILE" \
    --neo4j-uri "$NEO4J_URI" \
    --neo4j-user "$NEO4J_USER" \
    --neo4j-pass "$NEO4J_PASS"
if [[ $? -ne 0 ]]; then
    echo "Error: Neo4j load failed."
    exit 1
fi
echo "Done."

# ----------------------------------------------------------------------
# Step 3: Generate question_answers.json
# ----------------------------------------------------------------------
echo "Step 3: Generating question_answers.json from $TEMPLATE_FILE and the loaded graph ..."

# Build base command
CMD_QUESTION=("python" "$GEN_QUESTION_ANSWER_PY" \
    "-t" "$TEMPLATE_FILE" \
    "-d" "$OUTPUT_DIR/instances.json" \
    "-o" "$OUTPUT_DIR/question_answers.json" \
    "-n" "$NUM_PER_TYPE" \
    "--seed" "$SEED" \
    "--neo4j-uri" "$NEO4J_URI" \
    "--neo4j-user" "$NEO4J_USER" \
    "--neo4j-pass" "$NEO4J_PASS")

# Check for type count overrides: 1) CLI --type-counts, 2) specify.yaml question_types
if [[ -n "$TYPE_COUNTS" ]]; then
    # CLI takes highest precedence
    CMD_QUESTION+=("--type-counts" "$TYPE_COUNTS")
    echo "Using CLI --type-counts override."
elif [[ -f "$SPEC_FILE" ]]; then
    # Extract question_types from specify.yaml using Python
    TYPE_COUNTS_JSON=$(python -c "
import yaml, json
try:
    with open('$SPEC_FILE') as f:
        data = yaml.safe_load(f)
    qt = data.get('question_types')
    print(json.dumps(qt) if qt else '')
except Exception:
    print('')
" )
    if [[ -n "$TYPE_COUNTS_JSON" ]]; then
        CMD_QUESTION+=("--type-counts" "$TYPE_COUNTS_JSON")
        echo "Using question_types from $SPEC_FILE."
    fi
fi

# Execute question-answer generation
"${CMD_QUESTION[@]}"
if [[ $? -ne 0 ]]; then
    echo "Error: Question-answer generation failed."
    exit 1
fi
echo "Done."

# ----------------------------------------------------------------------
# Step 4: Generate table CSVs
# ----------------------------------------------------------------------
echo "Step 4: Generating table CSVs from $TABLE_SCHEMA_FILE and instances.json ..."
python "$GEN_TABLES_PY" -d "$OUTPUT_DIR/instances.json" -s "$TABLE_SCHEMA_FILE" -o "$OUTPUT_DIR/tables"
if [[ $? -ne 0 ]]; then
    echo "Error: Table generation failed."
    exit 1
fi
echo "Done."

# ----------------------------------------------------------------------
# Step 5: Assemble MMQA-format JSON + KG statistics
# ----------------------------------------------------------------------
echo "Step 5: Assembling MMQA-format JSON and KG statistics ..."
python "$ASSEMBLE_SELF_PY" -i "$OUTPUT_DIR" --kgschema "$SCHEMA_FILE" --table-schema "$TABLE_SCHEMA_FILE"
if [[ $? -ne 0 ]]; then
    echo "Error: Assembly step failed."
    exit 1
fi
echo "Done."

# ----------------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------------
echo ""
echo "Pipeline completed successfully."
echo "Output directory: $OUTPUT_DIR"
echo "  - instances.json"
echo "  - question_answers.json"
echo "  - tables/ (CSV files)"
echo "  - assembled_json/ (self.json, merged_self.json, question_type_stats.json)"
echo "  - stats.txt (KG statistics)"
echo ""

# Optional: list generated files
ls -l "$OUTPUT_DIR"
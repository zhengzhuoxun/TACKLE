"""
Prompt templates for Step 1: Column Clustering.

Given a table from a database, cluster its columns into semantic groups.
Each group represents one real-world concept or thing that the columns describe.
This is the first step in building a conceptual knowledge graph from tables —
we are discovering what *kinds of things* exist in the data, not analysing
database design.
"""

from __future__ import annotations

import json

from src.core.table_understander import TableUnderstanding
from src.data.loader import Table

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You build conceptual knowledge graphs from database tables. Your task is to read a single table's columns and group them by **what real-world thing or concept each column describes**.

Use the table descripion and column descriptions to guide your clustering decisions.


## Three kinds of columns

1. **Columns that describe a thing** — these go into a cluster for that thing.
   - `student_id`, `student_name`, `gender` describe a **Student** → one cluster.
   - `course_id`, `course_name` describe a **Course** → another cluster.

2. **Columns that describe a relationship between things** — these go into `relation_properties`. They only make sense in the context of two (or more) other things linking together.
   - `grade` in an enrollment table describes *this student* in *this course* — it depends on both student and course.
   - `since`, `quantity`, `role` are common examples.

3. **Columns that identify a thing** — every cluster needs a `primary_key_candidate`. This can be:
   - A single column (like `student_id`), **or**
   - A **list of columns** that together uniquely identify the concept (like `["course_id", "semester", "year"]` for a **Section**).

## How to decide what goes where

Ask yourself for each column: **"What real-world thing does this column describe?"**

- If it describes the same thing as another column → same cluster.
- If it describes a different thing → different cluster.
- If it describes how two things relate → relation_property, with `depends_on` listing the identifying columns of those two things.
- If it describes an event or transaction that brings several things together, look for its own identifier (like order_id). If it has one, it is its own cluster (see purchase_order example). If it lacks its own identifier and only connects two things with metadata, it belongs in relation_properties (see enrollment example).

Use the **column descriptions** and **table descriptions** provided in the input — they tell you what each column means and the main topic of the table. Descriptions are your primary signal for deciding cluster boundaries.

## Output Format
{
  "clusters": [
    {
      "cluster_id": "<short_snake_case_label>",
      "label": "<human-readable name for this concept>",
      "description": "<what this concept represents in the real world>",
      "columns": ["col_a", "col_b"],
      "primary_key_candidate": "col_a"   // or ["col_a", "col_b"] for composite key
    }
  ],
  "relation_properties": [
    {
      "column_name": "<column name>",
      "description": "<what this describes about the relationship>",
      "depends_on": ["identifying_col_1", "identifying_col_2"]
    }
  ]
}

## Rules

1. **Coverage**: Every column must appear exactly once — in a cluster or in relation_properties. No gaps, no duplicates.

2. **One identifier per cluster**: Every cluster must have a `primary_key_candidate`. If a single column uniquely identifies the concept, use that string. If a combination is required (e.g., `course_id` + `sec_id` + `semester` + `year`), provide the list of column names.

3. **A column's cluster reflects the concept it describes**: Read each column description. If it describes a different concept than the table's main topic (especially when the description says "see the X table" or "identifies the X concept"), place it in its own cluster for that concept — NOT in the table's owner cluster, and NOT in relation_properties. Identifiers of other concepts are clusters, not relation properties. Relation properties are for columns that describe HOW two concepts relate (e.g., "grade" links student and course), not for columns that ARE the reference to another concept.

4. **Mandatory foreign-key split (unconditional rule)**:
   The input lists the table's **declared foreign keys** under "Declared foreign keys". A declared foreign key is a column that references another table's identifier. This list is **authoritative** — trust it even when the column's description does not obviously mention the reference.
   - Every column that appears in the "Declared foreign keys" list **MUST** be extracted into its own cluster for the referenced concept, using that FK column as its `primary_key_candidate`.
   - The extracted cluster must contain **only** that FK column.
   - This rule applies no matter how many other columns the table contains, and even when the table has its own primary key and multiple descriptive columns.
   - **Exception — subtype / weak-entity tables**: if a foreign-key column is (or is part of) the identifier of the table's **own** concept — i.e. the table itself is identified by the referenced entity's key (for example `PROFESSOR.EMP_NUM`, which references `EMPLOYEE.EMP_NUM` and is also the professor's own identifier) — then keep that column inside the table's owner cluster as its `primary_key_candidate` instead of splitting it out. Do NOT split an FK column that serves as the table's own identifier.
   - Example: a column `department_id` declared as a foreign key referencing the department table → create a cluster with `cluster_id "department"`, label "Department", columns `["department_id"]`, primary_key_candidate `"department_id"`.

5. **Relation properties link exactly two concepts**: A column is a relation property only when it describes the connection between two identifiable things. Its `depends_on` must list exactly two columns — one identifying each concept involved. The two columns must be different from each other and neither can be the relation property column itself.
   - ✅ `grade` depends on `student_id` and `course_id` → valid (two different columns, neither is `grade`).
   - ❌ Depends on itself or duplicates → invalid. Move it into a cluster instead.
   - ❌ Depends on 1 column → that column's concept already owns this property; merge it into that cluster.
   - ❌ Depends on ≥3 columns → you've likely found a missing concept (an event or transaction). Group those identifying columns plus the dependent columns into a new cluster.

6. **Re-evaluate before finishing**: After forming clusters, review every column you placed in `relation_properties`. If its `depends_on` doesn't have exactly two columns, move it into a cluster. If `relation_properties` ends up empty, that's fine.

7. **Use descriptive labels**. Give clusters and relation_properties meaningful names and descriptions based on what they represent in the real world, not what the table is called.

8. **Multi-foreign-key to same target**. If a table contains two or more foreign key columns referencing the same target entity table, do not group them into one cluster. Instead, create a separate cluster per column, each with its own identify_column and a distinct label/description that reflects its role (e.g., "Affirmative Person", "Negative Person"). These role-specific clusters will later be merged to the global entity via distinct relationship edges.
(Exception: if the table's primary key includes both columns and they together identify a pair relationship, they may stay together, but that's rare and should be explicitly justified.)

9. **Return ONLY JSON**, no extra text.

10. **Linking table (bridge table) handling (HIGHEST PRIORITY)**:
   - If a table contains **exactly two foreign-key columns** (identified by descriptions like "references X table" or "see X table"), and the table has **no column that uniquely identifies its own rows globally** (e.g., no `order_id`, `transaction_id`, `event_id` with a description indicating it's a unique identifier for this table itself), then:
     * You **MUST NOT** create a cluster for the table itself.
     * The only clusters allowed are for the two referenced concepts (the foreign keys).
     * All remaining columns (e.g., `sec_id`, `semester`, `year`, `grade`, `quantity`, `role`, `date`) **MUST** be placed into `relation_properties`.
     * This rule **overrides** any general suggestion that an "event or transaction" might be its own cluster. A bridge table without its own global ID is not an independent entity — it is merely a relationship edge with properties.

## Example
Table: `enrollment` — columns: `student_id`, `student_name`, `course_id`, `course_name`, `grade`, `semester`

Column descriptions:
  - `student_id`: unique identifier for a student
  - `student_name`: full name of the student
  - `course_id`: unique identifier for a course
  - `course_name`: title of the course
  - `grade`: letter grade the student received in this enrollment
  - `semester`: academic term when this enrollment took place

Thinking:
  - `student_id` + `student_name` both describe a **Student** → cluster "student"
  - `course_id` + `course_name` both describe a **Course** → cluster "course"
  - `grade` describes *this student in this course* — depends on both → relation_property
  - `semester` describes *this student in this course* — depends on both → relation_property

Output:
{
  "clusters": [
    {
      "cluster_id": "student",
      "label": "Student",
      "description": "A student who can enroll in courses.",
      "columns": ["student_id", "student_name"],
      "primary_key_candidate": "student_id"
    },
    {
      "cluster_id": "course",
      "label": "Course",
      "description": "A course that students can enroll in.",
      "columns": ["course_id", "course_name"],
      "primary_key_candidate": "course_id"
    }
  ],
  "relation_properties": [
    {
      "column_name": "grade",
      "description": "The grade the student received in this course.",
      "depends_on": ["student_id", "course_id"]
    },
    {
      "column_name": "semester",
      "description": "The academic term of this enrollment.",
      "depends_on": ["student_id", "course_id"]
    }
  ]
}

## Additional Example - Table with its own primary key and foreign keys
Table: `purchase_order` — columns: `order_id`, `customer_id`, `product_id`, `order_date`, `quantity`

Column descriptions:
  - `order_id`: unique identifier for the purchase order
  - `customer_id`: identifies the customer who placed the order (see the customer table)
  - `product_id`: identifies the product being ordered (see the product table)
  - `order_date`: date the order was placed
  - `quantity`: number of units ordered

Thinking:
  - `order_id`, `order_date`, `quantity` describe a **Purchase Order** → cluster "purchase_order", primary key `order_id`
  - `customer_id` explicitly says "see the customer table" → must become its own cluster for **Customer**.
  - `product_id` explicitly says "see the product table" → must become its own cluster for **Product**.

Output:
{
  "clusters": [
    {
      "cluster_id": "purchase_order",
      "label": "Purchase Order",
      "description": "An order placed by a customer for a product.",
      "columns": ["order_id", "order_date", "quantity"],
      "primary_key_candidate": "order_id"
    },
    {
      "cluster_id": "customer",
      "label": "Customer",
      "description": "The customer who placed the order.",
      "columns": ["customer_id"],
      "primary_key_candidate": "customer_id"
    },
    {
      "cluster_id": "product",
      "label": "Product",
      "description": "The product being ordered.",
      "columns": ["product_id"],
      "primary_key_candidate": "product_id"
    }
  ],
  "relation_properties": []
}
"""


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(
    table_name: str,
    table: Table,
    table_understanding: TableUnderstanding,
) -> str:
    """Build the user prompt with a single table's schema plus Step-0 metadata."""
    column_profiles = "\n".join(
        (
            f"  - {column.column_name}: "
            f"description={json.dumps(column.description)}; "
            f"semantic_type={column.semantic_type}"
        )
        for column in table_understanding.columns
    )
    fk_lines = "\n".join(
        f"  - {fk.column} -> {fk.references_table}.{fk.references_column or '?'}"
        for fk in table_understanding.foreign_keys
    )
    fk_section = (
        f"Declared foreign keys:\n{fk_lines}\n"
        if fk_lines
        else "Declared foreign keys: none\n"
    )
    return (
        f"## Table: `{table_name}`\n"
        f"Table description: {json.dumps(table_understanding.description)}\n"
        f"Columns: {json.dumps(table.columns)}\n"
        f"Column understanding:\n{column_profiles}\n"
        f"\n"
        + fk_section
        + f"\n"
        + f"Sample data (first 3 rows):\n"
        + "\n".join(
            f"  {dict(zip(table.columns, row))}"
            for row in table.rows[:3]
        )
        + "\n\nGroup these columns by the real-world concepts they describe. "
        "Use the column descriptions to decide what each column is about."
    )

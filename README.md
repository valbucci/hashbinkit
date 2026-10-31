# HashBinKit

Collection of utilities for hashing binary executables.

Relies on:

- [r2bagpipe](https://github.com/valbucci/r2bagpipe):
  for loading, parsing and iterating binary data.
- [hexio](https://github.com/valbucci/hexio):
  for reading/writing/printing hashes in hexadecimal form.

## Usage

Currently, it requires binary data to be pre-cached
with [Bin2ML](https://github.com/valbucci/bin2ml).
Therefore you should have two paths:

1. Your binary file / binaries folder
2. The directory containing the raw cached data

### TODO

Below is the intended functionality:

- [ ] 1. Produce function-level hashes for all artefacts:
  For the first implementation store everything in CSV files.
  Order does *not* matter in scalar and graph artefacts.
  Scalar and graph hashes are computed both by
  **frequency** (features weighted by occurence) and
  **presence** (each different feature is hashed once).
  - [ ] 1.1. Scalar artefacts
    - [ ] 1.1.1. N. Instructions
    - [ ] 1.1.2. N. Variables
      - [ ] arguments
      - [ ] locals
      - [ ] locals + arguments
    - [ ] 1.1.3. CFG data
      - [ ] N. CFG Blocks
      - [ ] N. CFG Edges
      - [ ] cyclomatic complexity
    - [ ] 1.1.4. Function call-graph
      - [ ] indegree
      - [ ] outdegree
      - [ ] outputs indegree  (oi)
      - [ ] inputs indegree   (ii)
      - [ ] outputs outdegree (oo)
      - [ ] inputs outdegree  (io)
  - [ ] 1.2. Iterable artefacts (frequency / presence)
    - [ ] 1.2.1. Assembly immediates
      - [ ] all
      - [ ] no references
      - [ ] no references + no values < 0x100
    - [ ] 1.2.2. Assembly mnemonic types
      - [ ] plain
      - [ ] canonicalised
    - [ ] 1.2.3. ESIL instructions
      - [ ] plain
      - [ ] stripped values
    - [ ] 1.2.4. References
      - [ ] read
      - [ ] write
      - [ ] execute
      - [ ] all
    - [ ] 1.2.5. Variable data types (stripped bitness)
      - [ ] arguments
      - [ ] locals
      - [ ] arguments + locals
  - [ ] 1.3. Graph artefacts (frequency / presence)
    - [ ] 1.3.1. Control Flow Graph (CFG)
      - [ ] traversal indexing (machoke one-hops)
      - [ ] reference labelling (r / w / x)
      - [ ] subgraphs adjacency (k-3, k-4, k-5)
      - [ ] indegree/outdegree (k-1, k-2, k-3, k-4, k-5)
    - [ ] 1.3.2. Function Call Graph (CG)
      - [ ] traversal indexing (machoke one-hops)
      - [ ] reference labelling (r / w)
      - [ ] subgraphs adjacency (k-3, k-4, k-5)
      - [ ] indegree/outdegree (k-1, k-2, k-3, k-4, k-5)
- [ ] 2. Storage solution for hashes results
  A large amount of data will be produced, raw storage is not an option.
  Processing this data will require efficient data management.
  This is a hashing library though so any storage solution should
  prioritise portability over query efficiency. Downstream apps
  could use HashBinKit to read the output and store it in their DB.
  - [ ] 2.1. Stream `FEATURE_TYPE, FUNC_ADDRESS, FEATURE_HASH` tuples
             with a Python iterator directly from cached binary artefacts.
  - [ ] 2.2. Object-Relational Model to manage parquet dataset.
    Using parquet for lightweight compression and hierarchical de-duplication.
    Hierarchy follows structure `FILE_HASH > FEATURE > FUNC_ADDRESS > HASH`
    - [ ] 2.2.1. Ingest CSV files into single parquet file.
    - [ ] 2.2.2. Cached artefacts processed into feature hashes stored
                 in parquet file.
    - [ ] 2.2.3. Stream `FEATURE_TYPE, FUNC_ADDRESS, FEATURE_HASH` tuples
                 with a Python iterator from parquet file.

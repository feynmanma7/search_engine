<h1>Search Engine</h1>

# Version

+ 0.1, for English pdf documents searching.

# Goal

1. Support English words searching, support multi-word query.

# Index

## Inverted Index

> word -> [doc_id]

## Process

1. Iterate over documents, generate a doc_id, yield word.

2. For each word, add doc_id to the `inverted-index` only `once` (use a dict to control).

3. Flush the `sorted-inverted-index` and the dict of `{doc_id:doc_path}` to disk.

# Query

1. Get the `[doc_id]` of each of word in the `query`.

2. Compute the join of the `[doc_id]`.

3. Return the `[doc_path]`


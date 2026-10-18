# Compatibility Includes

Supported implementations now live in `source/contrib/nmfd-operations/include`.
The matching headers here are forwarding includes for historical paths, not a
second implementation. Put new operations and tests in the submodule.

Glued vectors, spaces, and matrix operators now use the standalone implementation.
The `common/glued_vector_*` headers preserve the historical `scfd::linspace` names
as aliases, without maintaining a second implementation.

The unfinished HYPRE adapters remain unchanged for reference.
Their copies in the standalone repository's `legacy/` directory are not supported
public headers. Migrate and test their project dependencies separately.

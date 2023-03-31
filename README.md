# Regiorust

Pour commencer il faut compiler le brol avec `maturin develop --release`.
Tu peux aussi faire `maturin publish --release` apres ca si tu veux.

Ensuite il suffit d'appeler le code rust comme ca:

```python
import regiorust as rr


# vertex    -> juste la matrice de float
# neighbors -> list d'adj
# edges     -> liste de tuples (source, dest)
# k         -> ton k
# width     -> max width du mdd
# timeout   -> apres quoi on arrete
solution = rr.solve_regionalization(vertex, neighbors, edges, k, width, timeout)

print(solution.proved)
print(solution.h_tot)
print(solution.deleted_edges)
```
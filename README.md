# Regiorust

Pour compiler le truc, il suffit de le faire avec `cargo build --release`.
Une fois que c'est fini, l'executable `target/release/regiorust` sera créé.
Cet executable s'utilise comme ceci:

```
Solve regionalization problem with MDDs

Usage: regiorust [OPTIONS] --vertices <VERTICES> --neighbors <NEIGHBORS> --id2edges <ID2EDGES>

Options:
  -v, --vertices <VERTICES>    Path to a file containing a matrix of weighted attributes
  -n, --neighbors <NEIGHBORS>  Path to a file containing the adjacency list of the computed tree
  -i, --id2edges <ID2EDGES>    Path to a file containing a list of tuples (source, destination) representing the edges of the graph. Each line corresponds to one line. The line number is the identifier of that edge
  -k, --k <K>                  The number of desired regions in the end [default: 3]
  -w, --w <W>                  The maximum width allowed for any MDD that is compiled [default: 10]
  -t, --timeout <TIMEOUT>      The maximum duration to solve the problem instance [default: 60]
  -h, --help                   Print help
  -V, --version                Print version
```

## Exemple d'utilisation

```
> ./target/release/regiorust -v example_vertex.txt -n example_neighbors.txt -i example_edges.txt
"proved", 0, "[(0, 1), (0, 2)]"
```

Le resultat du truc est une ligne qu'on peut ajouter dans un CSV. Cette ligne
reprend les infos suivantes (dans l'ordre): 
1. status -> ('proved', 'current-best', ou 'no-solution')
2. h_tot  -> l'heterogeneite totale
3. la liste des edges qui ont ete supprimées.

## Contenu des fichiers

Comme tu l'as vu plus haut, l'executable a besoin d'au mois trois parametres
qui correspondent a des fichiers. Ceux-ci ont le format suivant (exemples):

### verctices

L'attribut 0 du noeud 0 vaut 10. L'attribut 2 du noeud 1 vaut 12. etc..
```
10.00 9.00 7.00 0.0
5.00 3.25 12.4 9.00
23.00 6.0 14.3 8.00
```

### neighbors
Le noeud 0 a deux fils: 1 et deux. Les autres noeuds n'ont aucun fils.
```
1 2


```

### edges
L'edge 0 correspond a (0-1), l'edge 1 correspond a (0-2)
```
0 1
0 2
```

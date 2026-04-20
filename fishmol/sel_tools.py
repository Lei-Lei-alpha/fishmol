import itertools
from recordclass import make_dataclass
from fishmol.data import val_R

def cluster(atoms, mic=False):
    """
    Cluster atoms into molecules via covalent-radius bonding graph.
    Returns a list of Molecule dataclass instances; access .formula and .at_idx.
    """
    molecule = make_dataclass("Molecule", "formula at_idx")
    atoms.wrap_pos()

    symbs = atoms.symbs
    n = len(symbs)

    # Precompute per-atom covalent radii to avoid N² dict lookups in inner loop
    radii = [val_R[s] for s in symbs]

    # Build adjacency list from all bonded pairs
    adj = [set() for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        if atoms.dist(i, j, mic=mic) < 1.05 * (radii[i] + radii[j]):
            adj[i].add(j)
            adj[j].add(i)

    # Connected components via iterative DFS — O(N + E)
    visited = [False] * n
    mols_indices = []
    for start in range(n):
        if not visited[start]:
            component = []
            stack = [start]
            visited[start] = True
            while stack:
                u = stack.pop()
                component.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            mols_indices.append(sorted(component))

    # Build chemical formula for each molecule (alphabetical element order)
    final_mols = []
    for mol in mols_indices:
        counts = {}
        for s in symbs[mol]:
            counts[s] = counts.get(s, 0) + 1
        formula = "".join(
            s + (str(counts[s]) if counts[s] > 1 else "")
            for s in sorted(counts)
        )
        final_mols.append(molecule(formula, mol))

    return final_mols

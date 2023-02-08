import itertools as itr

# === IMPORTS: THIRD-PARTY ===
import causaldag as cd


class IntegerProgram:
    def __init__(
        self,
        true_graph: cd.DAG,
        true_ix2target: dict,
        est_ix2target: dict,
        solver: str = "gurobi"
    ) -> None:
        self.true_graph = true_graph
        self.true_ix2target = true_ix2target
        self.est_ix2target = est_ix2target
        self.solver = solver
        self.min_arcs = true_graph.arcs

    def create_model_gurobi(self):
        import gurobipy as gp
        from gurobipy import quicksum

        model = gp.Model("minimum")
        d = self.true_graph.nnodes
        K = len(self.true_ix2target)

        # === CREATE THE DECISION VARIABLES
        indicators = dict()
        for i, j in itr.product(range(d), repeat=2):
            indicators[(i, j)] = model.addVar(vtype="B", name=f"A_{i}{j}")
        
        # === CREATE THE CONSTRAINTS
        # A is a permutation
        for i in range(d):
            inds1 = [indicators[(i, j)] for j in range(d)]
            inds2 = [indicators[(j, i)] for j in range(d)]
            model.addConstr(quicksum(inds1) == 1, f"left{i}")
            model.addConstr(quicksum(inds2) == 1, f"right{i}")

        # A is consistent with G
        for j in range(d):
            for i1, i2 in self.min_arcs:
                inds_i1 = [indicators[(i1, j_)] for j_ in range(j)]
                inds_i2 = [indicators[(i2, j_)] for j_ in range(j)]
                model.addConstr(quicksum(inds_i1) - quicksum(inds_i2) >= 0, f"sum_{i1,}{i2},{j}")

        # === CREATE THE OBJECTIVE
        weight_terms = []
        for k in range(K):
            targ_true = self.true_ix2target[k]
            targ_est = self.est_ix2target[k]
            weight_terms.append(indicators[(targ_true, targ_est)])

        model.setObjective(quicksum(weight_terms), gp.GRB.MAXIMIZE)

        return model, indicators

    def solve_gurobi(self):
        model, indicators = self.create_model_gurobi()
        model.optimize()
        match = self.gurobi_solution2perm(indicators)
        return match

    def gurobi_solution2perm(self, indicators):
        d = len(self.true_ix2target)
        perm = [0] * d
        for i, j in itr.product(range(d), repeat=2):
            ind = indicators[(i, j)]
            if ind.X == 1:
                perm[i] = j
        return perm

    def solve(self):
        if self.solver == "gurobi":
            return self.solve_gurobi()
        else:
            raise ValueError
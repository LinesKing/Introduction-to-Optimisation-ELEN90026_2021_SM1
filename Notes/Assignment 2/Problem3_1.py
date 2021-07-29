import pyomo.environ as pyo

## Problem 3.(1)
# A quadratic program
model = pyo.AbstractModel()
model.name = 'QP'

# Note variables
model.x = pyo.Var(range(2))

# Define model and constrains
def QP(model):
    return model.x[0] ** 2 + 2 * model.x[1] ** 2 - model.x[0] * model.x[1] - model.x[0]


def ineqconstr1(model):
    return model.x[0] + 2 * model.x[1] <= -2


def ineqconstr2(model):
    return model.x[0] - 4 * model.x[1] <= -3


def ineqconstr3(model):
    return 5 * model.x[0] + 76 * model.x[1] <= 1


model.obj = pyo.Objective(rule=QP, sense=pyo.minimize)
model.ineqconstr1 = pyo.Constraint(rule=ineqconstr1)
model.ineqconstr2 = pyo.Constraint(rule=ineqconstr2)
model.ineqconstr3 = pyo.Constraint(rule=ineqconstr3)

# Create an instance of the problem
qp = model.create_instance()

# Lagrange multipliers (dual variables)
qp.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

# Define solver
opt = pyo.SolverFactory('ipopt')

# Show results
results = opt.solve(qp)


def display_lagrange(instance):
    # display all duals
    print("Duals")
    for c in instance.component_objects(pyo.Constraint, active=True):
        print("   Constraint", c)
        for index in c:
            print("      ", index, -instance.dual[c[index]])


display_lagrange(qp)


def disp_soln(instance):
    output = []
    for v in instance.component_data_objects(pyo.Var, active=True):
        output.append(pyo.value(v))
        print(v, pyo.value(v))
    print(instance.obj, pyo.value(instance.obj))
    output.append(pyo.value(instance.obj))
    return output


disp_soln(qp)

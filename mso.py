from cddd.inference import InferenceModel
import random
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Descriptors import qed
import numpy as np
from scipy import interpolate

infer_model = InferenceModel()


def initialize_particles(init_smiles, num_particles):
    smiles = []
    # particles = []
    for i in range(num_particles):
        index = random.randint(0, len(init_smiles)-1)
        smile = init_smiles[index]
        smiles.append(smile)
    particles = infer_model.seq_to_emb(smiles)
    return smiles, particles


def desirability_calc(desirability, xnew):
    x = [point['x'] for point in desirability]
    y = [point['y'] for point in desirability]
    f = interpolate.interp1d(x, y)
    ynew = f(xnew)
    return ynew


def getScore(desirability_curve, smile):
    unscaled_score = qed(Chem.MolFromSmiles(smile))
    desirability_score = desirability_calc(desirability_curve, unscaled_score)
    score_wt = 100
    scaled_score = desirability_score * score_wt
    return scaled_score


def init_swarm(init_smiles, num_particles):
    swarm = {}
    smiles, particles = initialize_particles(init_smiles, num_particles)
    swarm["num_particles"] = num_particles
    swarm["smiles"] = smiles
    swarm["particles"] = particles
    velocities = []
    for i in range(len(particles)):
        v = []
        for j in range(len(particles[i])):
            v.append(random.uniform(-0.6, 0.6))
        velocities.append(v)
    swarm["velocities"] = velocities
    swarm["personal_best_molecule"] = [p for p in particles]
    swarm["personal_best_score"] = [-1 for _ in range(len(particles))]
    swarm["desirability_curve"] = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
    swarm["overall_best_score"] = getScore(
        swarm["desirability_curve"], smiles[0])
    swarm["overall_best_molecule"] = particles[0]
    swarm["inertia_weight"] = 0.9
    swarm["c1"] = 2
    swarm["c2"] = 2
    return swarm


def update_velocities(swarm):
    for i in range(swarm["num_particles"]):
        n = len(swarm["particles"][0])
        r1 = swarm["c1"] * np.random.uniform(0, 1, n)
        r2 = swarm["c1"] * np.random.uniform(0, 1, n)

        # update velocity
        temp1 = swarm["personal_best_molecule"][i] - swarm["particles"][i]
        temp2 = swarm["overall_best_molecule"] - swarm["particles"][i]

        velocities_v1 = [swarm["inertia_weight"] * swarm["velocities"][i][k]
                         for k in range(n)]
        velocities_v2 = [r1[k] * temp1[k]
                         for k in range(n)]
        velocities_v3 = [r2[k] * temp2[k]
                         for k in range(n)]

        swarm["velocities"][i] = [velocities_v1[k] +
                                  velocities_v2[k] + velocities_v3[k] for k in range(n)]

        # update position
        x = swarm["particles"][i] + swarm["velocities"][i]

        x = np.clip(x, -1., 1.)
        swarm["particles"][i] = x
        swarm["smiles"][i] = infer_model.emb_to_seq(x)

        return swarm


def run(swarm, epochs):
    for e in range(epochs):
        # for each particle
        print("Epoch: ", e)
        for i in range(swarm["num_particles"]):
            # calculate qed score for each particle
            try:
                s = getScore(swarm["desirability_curve"], swarm["smiles"][i])
            except:
                pass

            # if score better than personal best, update personal best
            if s > swarm["personal_best_score"][i]:
                swarm["personal_best_score"][i] = s

            # if score better than overall best, update overall best
            if s > swarm["overall_best_score"]:
                swarm["overall_best_score"] = s
                swarm["overall_best_molecule"] = swarm["particles"][i]

        # update inertia weight (decreases every iteration)
        swarm["inertia_weight"] -= swarm["inertia_weight"]/100

        swarm = update_velocities(swarm)

        print("Best Molecule:", infer_model.emb_to_seq(
            swarm["overall_best_molecule"]))
        print("Best Score:", swarm["overall_best_score"])

    print("Overall Best Molecule:", infer_model.emb_to_seq(
        swarm["overall_best_molecule"]))
    print("Overall Best Score:", swarm["overall_best_score"])


if __name__ == '__main__':
    # init_smiles = [
    #     "c1ccccc1", "Cc1cccc(OCC(O)CNC(C)(C)C)c1C", "O=CC1CO1"]
    init_smiles = ["c1ccccc1"]
    num_particles = 100
    swarm = init_swarm(init_smiles, num_particles)
    epochs = 100
    run(swarm, epochs)

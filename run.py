import pm4py
import os
from SPN_Simulator.SPN_Simulator import StochasticPetriNetSimulator
from utils.train_utils import splitRealLog, addStartEndEvents, addStartEndTransitions
from utils.createDataset import CreateDataset
from utils.Discriminator import Discriminator
from utils.repair_utils import updateModel
import pickle
import json
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()

import random
random.seed(72)

parser.add_argument('--case_study', type=str, default='Purchasing')
parser.add_argument('--max_number_iterations', type=int, default=15)
parser.add_argument('--show_shap', type=bool, default=False)
parser.add_argument('--save_shap', type=bool, default=True)

args = parser.parse_args()

case_study = args.case_study
save_shap = args.save_shap
show_shap = args.show_shap
max_number_iterations = args.max_number_iterations

log_real = pm4py.read_xes(f'data/{case_study}/log.xes')
log_real = pm4py.filter_event_attribute_values(log_real, 'lifecycle:transition', ['complete'], level="event", retain=True)
log_real = addStartEndEvents(log_real)

if 'logTrain.xes' in os.listdir(f'data/{case_study}') and 'logTrain.xes' in os.listdir(f'data/{case_study}') and 'logTrain.xes' in os.listdir(f'data/{case_study}'):
    real_train = pm4py.read_xes(f'data/{case_study}/logTrain.xes')
    real_val = pm4py.read_xes(f'data/{case_study}/logVal.xes')
    real_test = pm4py.read_xes(f'data/{case_study}/logTest.xes')
else:
    real_train, real_val, real_test = splitRealLog(log_real, split_size = (0.6, 0.2, 0.2), split_temporal = True, save_to=f'data/{case_study}')

try:
    n_exp = max([int(d.split('_')[-1]) for d in os.listdir(f'data/{case_study}') if d[:3]=='exp']) + 1
except:
    n_exp = 1

if case_study not in os.listdir('plots'):
    os.mkdir(f'plots/{case_study}')

os.mkdir(f'data/{case_study}/exp_{n_exp}')
os.mkdir(f'data/{case_study}/exp_{n_exp}/it_0')
os.mkdir(f'plots/{case_study}/exp_{n_exp}')

net, initial_marking, final_marking = pm4py.read_pnml(f'data/{case_study}/diagram_0.pnml')
net, initial_marking, final_marking = addStartEndTransitions(net, initial_marking, final_marking)
pm4py.write_pnml(net, initial_marking, final_marking, f'data/{case_study}/exp_{n_exp}/it_0/diagram_0.pnml')

train_accuracy = []
val_accuracy = []

it_n = 0
graph_is_updated = True

while it_n <= max_number_iterations and graph_is_updated:
    net, initial_marking, final_marking = pm4py.read_pnml(f'data/{case_study}/exp_{n_exp}/it_{it_n}/diagram_{it_n}.pnml')
    Simulator = StochasticPetriNetSimulator(net, initial_marking, final_marking, log=real_train)

    log_sim = Simulator.simulate(len(real_train) + len(real_val))
    pm4py.write_xes(log_sim, f'data/{case_study}/exp_{n_exp}/it_{it_n}/sim_{it_n}.xes')

    data_creator = CreateDataset(real_train, real_val, log_sim)

    discriminator = Discriminator(data_creator, catboost_iterations=100, catboost_depth=3, catboost_lr=1e-1)
    discriminator.fit()

    acc_train, acc_val = discriminator.eval_accuracy()
    print(f'It. {it_n}: train_accuracy {round(acc_train, 4)} --- val_accuracy {round(acc_val, 4)}')
    train_accuracy.append(acc_train)
    val_accuracy.append(acc_val)

    # reccomendations
    if save_shap:
        reccomendations = discriminator.reccomendations(show=show_shap, save_to=f'plots/{case_study}/exp_{n_exp}/shap_{it_n}.png')
    else:
        reccomendations = discriminator.reccomendations(show=show_shap)

    print(reccomendations)
    net, initial_marking, final_marking = pm4py.read_pnml(f'data/{case_study}/exp_{n_exp}/it_{it_n}/diagram_{it_n}.pnml')
    net, graph_is_updated = updateModel(net, initial_marking, final_marking, reccomendations)
    pm4py.view_petri_net(net, initial_marking, final_marking)

    os.mkdir(f'data/{case_study}/exp_{n_exp}/it_{it_n+1}')
    pm4py.write_pnml(net, initial_marking, final_marking, f'data/{case_study}/exp_{n_exp}/it_{it_n+1}/diagram_{it_n+1}.pnml')

    file = open(f"data/{case_study}/exp_{n_exp}/it_{it_n}/recc_{it_n}.pickle", "wb")
    pickle.dump((reccomendations), file)
    file.close()
    
    it_n += 1


history = {'train_accuracy': train_accuracy, 'validation_accuracy': val_accuracy}
file = open(f"data/{case_study}/exp_{n_exp}/history.json", "w")
json.dump(history, file)
file.close()


plt.figure()
plt.plot(train_accuracy, 'o-', label='train accuracy')
plt.plot(val_accuracy, 'o-', label='validation accuracy')
plt.title('Model Accuracy per iteration')
plt.legend()
plt.savefig(f'plots/{case_study}/exp_{n_exp}/history.png')
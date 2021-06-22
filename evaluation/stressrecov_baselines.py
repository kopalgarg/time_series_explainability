import os
import argparse
from pdb import set_trace
import torch
import numpy as np
import seaborn as sns; sns.set()
import pickle as pkl
import time
import sys
import wandb

sys.path.append('/Users/kopalgarg/Documents/GitHub/time_series_explainability/')
from matplotlib import rc, rcParams
rc('font', weight='bold')
from matplotlib import rc, rcParams
rc('font', weight='bold')

from TSX.utils import load_simulated_data, train_model_rt, compute_median_rank, train_model_rt_binary, \
    train_model_multiclass, train_model, load_data
from TSX.models import StateClassifier, RETAIN, EncoderRNN, ConvClassifier, StateClassifierMIMIC

from TSX.generator_modified import JointFeatureGenerator, JointDistributionGenerator
from TSX.explainers import RETAINexplainer, FITExplainer, IGExplainer, FFCExplainer, \
    DeepLiftExplainer, GradientShapExplainer, AFOExplainer, FOExplainer, SHAPExplainer, \
    LIMExplainer, CarryForwardExplainer, MeanImpExplainer, FITExplainer_moving_window
from sklearn import metrics

# b FITExplainer_moving_window.attribute



if __name__ == '__main__':
    np.random.seed(1234)
    parser = argparse.ArgumentParser(description='Run baseline model for explanation')
    parser.add_argument('--explainer', type=str, default='fit', help='Explainer model')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train_gen', action='store_true')
    parser.add_argument('--generator_type', type=str, default='history')
    parser.add_argument('--out_path', type=str, default='./output/')
    parser.add_argument('--mimic_path', type=str)
    parser.add_argument('--binary', action='store_true', default=False)
    parser.add_argument('--gt', type=str, default='true_model', help='specify ground truth score')
    parser.add_argument('--cv', type=int, default=0, help='cross validation')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 10
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')
    activation = torch.nn.Softmax(-1)
    output_path = args.out_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    feature_size = 3
    data_path = './data/n_deep_rmssd_hr_average_score'
    data_type='spike'
    n_classes = 5
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plot_path = os.path.join('./plots/%s' % args.data)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)


    _, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=batch_size, datapath=data_path,
                                                                         percentage=0.8, data_type=data_type, cv=args.cv)
    
    #       Multi-class
    model = StateClassifierMIMIC(feature_size=feature_size, n_state=n_classes, hidden_size=128,rnn='GRU')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    train_model_multiclass(model=model, train_loader=train_loader, valid_loader=test_loader,
                        optimizer=optimizer, n_epochs=25, device=device, experiment='model', data=args.data,num=5,
                        loss_criterion=torch.nn.CrossEntropyLoss(),cv=args.cv)

    #       Binary-class
    #model = StateClassifierMIMIC(feature_size=feature_size, n_state=n_classes, hidden_size=128,rnn='GRU')
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    #train_model_rt(model=model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, n_epochs=50,
    #                           device=device, experiment='model', data='temperature_delta_daily_control_rem_hr_average_scorebin',cv=args.cv)

    import pdb; pdb.set_trace()

generator = JointFeatureGenerator(feature_size, hidden_size=feature_size * 3, data=args.data)

#       FIT
explainer = FITExplainer(model)
explainer.fit_generator(generator, train_loader, valid_loader,cv=args.cv)

#       AFO
# explainer = AFOExplainer(model, train_loader)

importance_scores = []
ranked_features=[]
n_samples = 1

for x, y in test_loader:
    model.train()
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    _, n_features, t_len = x.shape
    t0 = time.time()
    moving_window = 3
    import pdb; pdb.set_trace()
    score = explainer.attribute(x, y if args.data=='n_deep_rmssd_hr_average_score' else y[:, -1].long())
    #ranked_feats = {}
    #for t in range(0, t_len):
    # ranked_feats["ranked_feats{0}".format(t+1)]= np.array([((-(score.get("score{0}".format(t+1))[n])).argsort(0).argsort(0) + 1) for n in range(x.shape[0])]) 
    ranked_feats = np.array([((-(score[n])).argsort(0).argsort(0) + 1) for n in range(x.shape[0])])
    importance_scores.append(score)
    ranked_features.append(ranked_feats)    

with open(os.path.join(output_path, '%s_test_importance_scores_%d.pkl' % (args.explainer, args.cv)), 'wb') as f:
    pkl.dump(importance_scores, f, protocol=pkl.HIGHEST_PROTOCOL)

with open(os.path.join(output_path, '%s_test_ranked_scores.pkl' % args.explainer), 'wb') as f:
    pkl.dump(ranked_features, f, protocol=pkl.HIGHEST_PROTOCOL)
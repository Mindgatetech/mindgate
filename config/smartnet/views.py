import datetime
import hashlib
import json
from joblib import dump
from . import models
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.db.models import Count
from django.conf import settings
from cpanel import models as cm
from mindnet import models as mnm
import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, KFold, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost.sklearn import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import importlib

def result_converter(res):
    results = res
    import numpy as np
    from prettytable import PrettyTable

    for data_key in results.keys():
        print(data_key)
        myTable = PrettyTable(["Pipe", "S0", "S1", "S2", "S3", "S4", "Average"])
        myTable.title = 'Results for ' + data_key
        for preprocess_key in results[data_key].keys():
            print(preprocess_key)
            for scaler_key in results[data_key][preprocess_key].keys():
                print(scaler_key)
                for framework_key in results[data_key][preprocess_key][scaler_key].keys():
                    print(framework_key)
                    for model_key in results[data_key][preprocess_key][scaler_key][framework_key].keys():
                        print(model_key)
                        re = np.array(
                            results[data_key][preprocess_key][scaler_key][framework_key][model_key]['test_accuracy'])
                        avg = re.mean()
                        re = np.append(re, avg)

                        s = data_key + ' -> ' + preprocess_key + ' -> ' + scaler_key + ' -> ' + framework_key + ' -> ' + model_key
                        re = re.tolist()
                        re.insert(0, s)
                        myTable.add_row(re)
    return myTable
def import_test(request):
    function_string = 'media.Ai_models.my_ai_model.my_model'
    mod_name, func_name = function_string.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    clf = func()
    print(clf)
    return HttpResponse("OK dude :)")
def hook_test(request):
    print("Hello I'm Scheduled Hook function")
    objs = models.Result.objects.filter(processed=False)
    import numpy as np
    from prettytable import PrettyTable
    for obj in objs:
        results = obj.results
        print(type(results))
        results = results.replace("'", "\"")
        results = json.loads(results)

        #print(results)
        #return HttpResponse("OK")

        for data_key in results.keys():
            print(data_key)
            ATable = PrettyTable(["Pipe", "S0", "S1", "S2", "S3", "S4", "Average"])
            ATable.title = 'Accuracy Results on Dataset_group: ' + data_key
            PTable = PrettyTable(["Pipe", "S0", "S1", "S2", "S3", "S4", "Average"])
            PTable.title = 'Precision Results on Dataset_group: ' + data_key
            RTable = PrettyTable(["Pipe", "S0", "S1", "S2", "S3", "S4", "Average"])
            RTable.title = 'Recall Results on Dataset_group: ' + data_key
            FTable = PrettyTable(["Pipe", "S0", "S1", "S2", "S3", "S4", "Average"])
            FTable.title = 'F1 Results on Dataset_group: ' + data_key
            for preprocess_key in results[data_key].keys():
                print(preprocess_key)
                for scaler_key in results[data_key][preprocess_key].keys():
                    print(scaler_key)
                    for framework_key in results[data_key][preprocess_key][scaler_key].keys():
                        print(framework_key)
                        for model_key in results[data_key][preprocess_key][scaler_key][framework_key].keys():
                            print(model_key)
                            s =  preprocess_key + '->' + scaler_key + '->' + framework_key + '->' + model_key
                            re = np.array(
                                results[data_key][preprocess_key][scaler_key][framework_key][model_key][0])
                            avg = re.mean()
                            re = np.append(re, avg)
                            re = re.tolist()
                            re.insert(0, s)
                            ATable.add_row(re)
                            re = np.array(
                                results[data_key][preprocess_key][scaler_key][framework_key][model_key][1])
                            avg = re.mean()
                            re = np.append(re, avg)
                            re = re.tolist()
                            re.insert(0, s)
                            PTable.add_row(re)
                            re = np.array(
                                results[data_key][preprocess_key][scaler_key][framework_key][model_key][2])
                            avg = re.mean()
                            re = np.append(re, avg)
                            re = re.tolist()
                            re.insert(0, s)
                            RTable.add_row(re)
                            re = np.array(
                                results[data_key][preprocess_key][scaler_key][framework_key][model_key][3])
                            avg = re.mean()
                            re = np.append(re, avg)
                            re = re.tolist()
                            re.insert(0, s)
                            FTable.add_row(re)
        results = dict()
        results = dict(Accuracy=ATable.get_csv_string(),Precision=PTable.get_csv_string(),
                   Recall=RTable.get_csv_string(), F1=FTable.get_csv_string())
        result_id = hashlib.shake_256(str(datetime.datetime.now()).encode()).hexdigest(5)
        '''for result_key in results.keys():
            save_path = 'media/Smartnet_results/'+data_key+'_'+result_key+'.csv'
            pd.DataFrame([x.replace('\r', '').split(',') for x in results[result_key].split("\n")[1:-1]],
                         columns=[x.replace('\r', '').split(',') for x in results[result_key].split("\n")][0])'''
        models.Result.objects.create(related_result=obj, result_id=result_id, results=results, processed=True)
    return HttpResponse("OK OK")
def hyperparameter_get(arg):
    hyperparameter = dict()
    for _, item in enumerate(arg.split(",")):
      tmp = item.split(":")
      hyperparameter[tmp[0]] = tmp[1]
    return hyperparameter
def Scheduled_test(request):

    value_list = mnm.Dataset.objects.filter(private=False).values_list(
        'metadata', flat=True
    ).distinct()
    print(value_list)
    group_by_value = dict()
    for value in value_list:
        group_by_value[value] = mnm.Dataset.objects.filter(private=False, metadata=value)
    print(group_by_value)
    data_path_by_group = dict()
    for i, key in enumerate(group_by_value.keys()):
        lis = list()
        for item in group_by_value[key]:
            efp = item.extracted_file_path.split(',')
            for p in efp:
                lis.append(p)
        data_path_by_group[i] = lis
    print(data_path_by_group)
    preprocesses = mnm.Preprocess.objects.filter(private=False)
    for preprocess in preprocesses:
        hyperparameter = hyperparameter_get(preprocess.hyperparameters)
        print(hyperparameter)
    value_list = mnm.AiModel.objects.filter(private=False).values_list(
        'framework', flat=True
    ).distinct()
    group_by_value = dict()
    for value in value_list:
        group_by_value[value] = mnm.AiModel.objects.filter(private=False, framework=value)
    print(group_by_value)
    return HttpResponse("OK!")
def Scheduled():
    print("Hello I'm Scheduled function")
    def read_raw_data(subject, path, eog, event_id, filter, freq, montage_type, timing, on_missing):
        raw = mne.io.read_raw_gdf(path, eog=eog, preload=True, verbose=False)
        '''if montage_type is not 'None':
            montage = mne.channels.make_standard_montage(montage_type)
            raw.set_montage(montage)'''
        if freq is not None:
            raw.filter(freq[0], freq[1], fir_design="firwin", skip_by_annotation="edge")
        raw.pick_types(meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        events, _ = mne.events_from_annotations(raw)
        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            proj=True,
            baseline=None,
            tmin=timing[0],
            tmax=timing[1],
            preload=True,
            on_missing=on_missing,
        )
        epochs_data = epochs.get_data(copy=True)
        labels = epochs.events[:, -1] - event_id[0]
        group = [subject] * epochs_data.shape[0]
        return epochs_data, labels, group
    value_list = mnm.Dataset.objects.filter(private=False).values_list(
        'metadata', flat=True
    ).distinct()
    group_by_value = dict()
    for value in value_list:
        group_by_value[value] = mnm.Dataset.objects.filter(private=False, metadata=value)
    data_path_by_group = dict()
    for key in group_by_value.keys():
        tmp_list = list()
        for item in group_by_value[key]:
            efp = item.extracted_file_path.split(',')
            for p in efp:
                tmp_list.append(p)
        data_path_by_group[key] = tmp_list
    # Group AIModels
    value_list = mnm.AiModel.objects.filter(private=False, granted=True).values_list(
        'framework', flat=True
    ).distinct()
    group_by_framework = dict()
    for value in value_list:
        group_by_framework[value] = mnm.AiModel.objects.filter(private=False, framework=value)


    # Preprocess
    preprocesses = mnm.Preprocess.objects.filter(private=False)
    # Scaler
    scalers = mnm.Scaler.objects.filter(private=False, granted=True)
    score_list = list()
    freq = [8,30]
    event_id = [7,8]
    timing = [0.5, 2.5]
    eog_channels = ['EOG:ch01','EOG:ch02','EOG:ch03']
    results = dict()
    result_id = hashlib.shake_256(str(datetime.datetime.now()).encode()).hexdigest(5)
    best_models_path = list()
    for key in data_path_by_group.keys():
        data = [read_raw_data(
            subject=subject, path=path, eog=eog_channels, event_id=event_id,
            filter=filter, freq=freq, montage_type=None, timing=timing, on_missing='warn')
            for subject, path in enumerate(data_path_by_group[key])]
        data_list = [x[0] for _, x in enumerate(data)]
        label_list = [x[1] for _, x in enumerate(data)]
        group_list = [x[2] for _, x in enumerate(data)]
        data_array = np.vstack(data_list)
        label_array = np.hstack(label_list)
        group_array = np.hstack(group_list)
        # Preprocess
        tmp_result_by_preprocess = dict()
        best_model_by_pipe = list()
        for preprocess in preprocesses:
            preprocess_key = preprocess.name
            hyperparameter = hyperparameter_get(preprocess.hyperparameters)
            csp = mne.decoding.CSP(n_components=int(hyperparameter['n_components']))
            data_array = csp.fit_transform(data_array, label_array)
            # Scaler
            tmp_result_by_scaler = dict()
            steps = list()
            for scaler in scalers:
                scaler_key = scaler.name
                #hyperparameter = hyperparameter_get(scaler.hyperparameters)
                if scaler.name == 'MinMaxScaler':
                    scaler = MinMaxScaler()
                    steps.append(('scaler', scaler))
                elif scaler.name == 'StandardScler':
                    scaler = StandardScaler()
                    steps.append(('scaler', scaler))
                else:
                    function_string = "media.Scalers." + str(
                        scaler.scaler.url.split('/')[-1].split('.')[0]) + ".my_scaler"
                    mod_name, func_name = function_string.rsplit('.', 1)
                    mod = importlib.import_module(mod_name)
                    func = getattr(mod, func_name)
                    steps.append(func())
                #######################################################################
                # End Scaler | Start Ai Model
                #######################################################################
                tmp_result_by_framework = dict()
                for key_f in group_by_framework.keys():
                    tmp_result_by_aimodel = dict()
                    for aimodel in group_by_framework[key_f]:
                        aimodel_key = aimodel.name
                        if aimodel.framework == 'Scikit-learn':
                            function_string = "media.Ai_models." + str(
                                aimodel.model.url.split('/')[-1].split('.')[0]) + ".my_model"
                            mod_name, func_name = function_string.rsplit('.', 1)
                            mod = importlib.import_module(mod_name)
                            func = getattr(mod, func_name)
                            steps.append(func())
                        elif aimodel.framework == 'Keras':
                            pass
                        else:
                            pass
                        #######################################################################
                        # End Ai Model | Start Pipeline
                        #######################################################################
                        pipe = Pipeline(steps)
                        pipe.fit(data_array, label_array)
                        # pop AiModel
                        steps.pop()
                        #### Pipeline End ####
                        scoring = ['accuracy',
                                   'precision',
                                    'recall',
                                    'f1'
                                   ]
                        score = cross_validate(pipe, data_array, label_array,
                                               cv=LeaveOneGroupOut(), scoring=scoring, n_jobs=1,
                                               return_estimator=True, groups=group_array)
                        score.pop('score_time')
                        score.pop('fit_time')
                        best_model_by_pipe.append((score.pop('estimator')[np.argmax(score['test_accuracy'])], score['test_accuracy'].mean()))
                        tmp_result_by_aimodel[aimodel_key] = [score['test_accuracy'].tolist(), score['test_precision'].tolist(),
                                                              score['test_recall'].tolist(),score['test_f1'].tolist()]
                        #score_list.append(score)
                    tmp_result_by_framework[key_f] = tmp_result_by_aimodel
                # pop Scaler
                steps.pop()
                tmp_result_by_scaler[scaler_key] = tmp_result_by_framework

            tmp_result_by_preprocess[preprocess_key] = tmp_result_by_scaler
        path_save = '/smartnet_models/' + result_id + datetime.datetime.now().strftime(
            '%Y_%m_%d_%H_%M_%S') + '.joblib'
        dump(best_model_by_pipe[np.argmax([m[1] for m in best_model_by_pipe])][0],
             filename=str(settings.BASE_DIR) + '/media' + path_save)
        best_models_path.append(path_save)
        results[key] = tmp_result_by_preprocess
    best_models_path = ','.join(bmp for bmp in best_models_path)
    models.Result.objects.create(result_id=result_id, results=str(results), best_models_path=best_models_path)
    return True
def Scheduled_hook(score_list):
    print("Hello I'm Scheduled Hook function")

def S():
    t = 'datetime.datetime.now()'
    print('Hi from SC function :', t)

def SCHook(t):
    print('Hi from SCHook function :', t)


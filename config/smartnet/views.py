import datetime
import hashlib

from . import models
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.db.models import Count
from cpanel import models as cm
from mindnet import models as mnm
import mne
import numpy as np
from sklearn.model_selection import cross_validate, KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost.sklearn import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import importlib

def import_test(request):
    function_string = 'media.Ai_models.my_ai_model.my_model'
    mod_name, func_name = function_string.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    clf = func()
    print(clf)
    return HttpResponse("OK dude :)")
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
    for i, key in enumerate(group_by_value.keys()):
        tmp_list = list()
        for item in group_by_value[key]:
            efp = item.extracted_file_path.split(',')
            for p in efp:
                tmp_list.append(p)
        data_path_by_group[i] = tmp_list
    # Group AIModels
    value_list = mnm.AiModel.objects.filter(private=False, granted=True).values_list(
        'framework', flat=True
    ).distinct()
    group_by_framework = dict()
    for value in value_list:
        group_by_framework[value] = mnm.AiModel.objects.filter(private=False, framework=value)
    aimodel_by_group = dict()

    # Preprocess
    preprocesses = mnm.Preprocess.objects.filter(private=False)
    # Scaler
    scalers = mnm.Scaler.objects.filter(private=False)
    score_list = list()
    freq = [8,30]
    event_id = [7,8]
    timing = [0.5, 2.5]
    eog_channels = ['EOG:ch01','EOG:ch02','EOG:ch03']

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
        for preprocess in preprocesses:
            hyperparameter = hyperparameter_get(preprocess.hyperparameters)
            csp = mne.decoding.CSP(n_components=int(hyperparameter['n_components']))
            data_array = csp.fit_transform(data_array, label_array)
            # Scaler
            steps = list()
            for scaler in scalers:
                #hyperparameter = hyperparameter_get(scaler.hyperparameters)
                if scaler.name == 'MinMaxScaler':
                    scaler = MinMaxScaler()
                    steps.append(('scaler', scaler))
                elif scaler.name == 'StandardScler':
                    scaler = StandardScaler()
                    steps.append(('scaler', scaler))
                for key_f in group_by_framework.keys():
                    for aimodel in group_by_framework[key_f]:
                        if aimodel.framework == 'Scikit-learn':
                            function_string = 'media.Ai_models.my_ai_model.my_model'
                            mod_name, func_name = function_string.rsplit('.', 1)
                            mod = importlib.import_module(mod_name)
                            func = getattr(mod, func_name)
                            steps.append(func())
                        elif aimodel.framework == 'Keras':
                            pass
                        else:
                            pass
                        #### Pipeline Start ####
                        pipe = Pipeline(steps)
                        pipe.fit(data_array, label_array)
                        # pop AiModel
                        steps.pop()
                        #### Pipeline End ####
                        scoring = ['accuracy', 'precision', 'recall', 'f1']
                        score = cross_validate(pipe, data_array, label_array,
                                               cv=GroupKFold(), scoring=scoring, n_jobs=1,
                                               return_estimator=True, groups=group_array)
                        score_list.append(score)
                # pop Scaler
                steps.pop()
    models.Result.objects.create(result_id=hashlib.shake_256(str(datetime.datetime.now()).encode()).hexdigest(5), result=score_list)
    return score_list
def Scheduled_hook(score_list):
    print("Hello I'm Scheduled Hook function")

def S():
    t = 'datetime.datetime.now()'
    print('Hi from SC function :', t)

def SCHook(t):
    print('Hi from SCHook function :', t)


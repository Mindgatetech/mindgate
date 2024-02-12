import time, requests, zipfile, mne, glob, scipy, io, urllib, base64, hashlib
from django.shortcuts import render, get_object_or_404
from django_q.tasks import async_task
from sklearn.model_selection import cross_validate, KFold, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost.sklearn import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from django.http import HttpResponse
from django.conf import settings
from datetime import datetime
from joblib import dump
import numpy as np
import importlib
from . import models


def test(request):
    models.Dataset.object
    return HttpResponse('Test OK!' + str(settings.BASE_DIR))

# Dataset Processor View
def dataset_processor(model):
    name    = model.name
    username= model.user.username
    url     = model.dataset_link

    path    ='./media/datasets_archived/'
    source  = path + name+'.zip'
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(source, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        finish_time = time.time()
    '''    path = 'media/datasets/' + datetime.now().strftime('%Y/%m/%d/%H%M%S/') + name
        with zipfile.ZipFile(source, 'r') as zip_ref:
            zip_ref.extractall(path)'''
    def renamefile(originalname, username):
      fn = originalname.split('.')
      name = ''.join(f for f in fn[:-1])
      extention = fn[-1]
      rename = username+'_'+name+'_'+datetime.now().strftime('%Y_%m_%d_%H_%M_%S.')+extention
      return rename
    zipdata = zipfile.ZipFile(source)
    zipinfos = zipdata.infolist()
    # iterate through each file
    extracted_file_path = list()
    for zipinfo in zipinfos:
        # This will do the renaming
        zipinfo.filename = renamefile(zipinfo.filename, username=username)
        extracted_file_path.append('media/datasets/'+zipinfo.filename)
        #zipdata.extract(zipinfo)
        zipdata.extract(zipinfo, 'media/datasets/')
    data_path = extracted_file_path[0]
    raw = mne.io.read_raw_gdf(data_path, verbose=False)
    ch_names = ','.join(i for i in raw.ch_names)
    model.extracted_file_path = ','.join(efp for efp in extracted_file_path)
    model.channels     = ch_names
    model.dataset_path = path
    model.metadata = hashlib.shake_256(ch_names.encode()).hexdigest(8)
    print(model.metadata)
    model.ready_to_use = True
    model.save()


def hyperparameter_get(arg):
    hyperparameter = dict()
    for _, item in enumerate(arg.split(",")):
      tmp = item.split(":")
      hyperparameter[tmp[0]] = tmp[1]
    return hyperparameter

# PipeJob Processor View
def pipjob_processing(model):
    efp             = model.dataset.extracted_file_path
    cv              = model.validationtechnique
    preprocess      = model.preprocess
    aimodel         = model.aimodel
    scaler          = model.scaler
    metric          = model.metric
    test_size = model.test_size
    tmin = model.tmin
    tmax = model.tmax
    timing = [tmin, tmax]
    event_id = [int(item) for item in model.event_id.split(',')]
    montage_type = model.montage_type
    filter = model.filter
    low_band = float(model.low_band)
    high_band = float(model.high_band)
    freq = [low_band, high_band]
    event_from = model.event_from
    on_missing = model.on_missing
    stim_channel = model.stim_channel
    eeg_channels = [item for item in model.eeg_channels.split(',')]
    eog_channels = [item for item in model.eog_channels.split(',')]
    exclude = model.exclude
    projection = model.projection
    baseline = model.baseline
    start_time = model.start_time
    end_time = model.end_time
    output_shape = model.output_shape
    trained_model = model.trained_model

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

    data_path = [fp for fp in efp.split(',')]
    '''if len(dataset_path) > 0:
        extention = dataset_path[0].split('.')[-1]
        print(extention)
        path = glob.glob(dataset_path + "/*." + extention)'''
        ######################
    data = [read_raw_data(
        subject=subject, path=path, eog=eog_channels, event_id=event_id,
        filter=filter, freq=freq, montage_type=montage_type, timing=timing, on_missing=on_missing)
        for subject, path in enumerate(data_path)]
    data_list = [x[0] for _, x in enumerate(data)]
    label_list = [x[1] for _, x in enumerate(data)]
    group_list = [x[2] for _, x in enumerate(data)]
    data_array  = np.vstack(data_list)
    label_array = np.hstack(label_list)
    group_array = np.hstack(group_list)

    if preprocess.name == 'CSP':
        hyperparameter = hyperparameter_get(preprocess.hyperparameters)
        csp = mne.decoding.CSP(n_components=int(hyperparameter['n_components']))
        data_array = csp.fit_transform(data_array, label_array)
    steps = list()
    if scaler.name == 'MinMaxScaler':
        scaler = MinMaxScaler()
        steps.append(('scaler', scaler))
    elif scaler.name == 'StandardScler':
        scaler = StandardScaler()
        steps.append(('scaler', scaler))
    else:
        if scaler.granted:
            function_string = "media.Scalers."+ str(scaler.scaler.url.split('/')[-1].split('.')[0]) + ".my_scaler"
            mod_name, func_name = function_string.rsplit('.', 1)
            mod = importlib.import_module(mod_name)
            func = getattr(mod, func_name)
            steps.append(func())
    if aimodel.granted:
        if aimodel.framework == 'Scikit-learn':
            function_string = "media.Ai_models."+ str(aimodel.model.url.split('/')[-1].split('.')[0]) + ".my_model"
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
    #### Pipeline End ####
    if cv.name == 'KFold':
        hyperparameter = hyperparameter_get(cv.hyperparameters)
        cv = KFold(n_splits=int(hyperparameter['n_splits']))
    elif cv.name == 'GroupKFold':
        hyperparameter = hyperparameter_get(cv.hyperparameters)
        cv = GroupKFold(n_splits=int(hyperparameter['n_splits']))
        group_array = None
    else:
        pass
    if metric.name == 'All':
        scoring = ['accuracy', 'precision', 'recall', 'f1']
    elif metric.name == 'Accuracy':
        scoring = ['accuracy', ]
    elif metric.name == 'Precision':
        scoring = ['precision', ]
    elif metric.name == 'Recall':
        scoring = ['recall', ]
    elif metric.name == 'F1':
        scoring = ['f1', ]
    else:
        if metric.granted:
            function_string = "media.Metrics." + str(metric.metric.url.split('/')[-1].split('.')[0]) + ".my_metric"
            mod_name, func_name = function_string.rsplit('.', 1)
            mod = importlib.import_module(mod_name)
            func = getattr(mod, func_name)
            scoring = func()
    score = cross_validate(pipe, data_array, label_array,
                           cv=cv,scoring=scoring, n_jobs=1,
                           return_estimator=True, groups=group_array)
    estimators = score.pop('estimator')
    path_save = '/models_trained/'+str(model.user.username)+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'.joblib'
    dump(estimators[0][1], filename=str(settings.BASE_DIR)+'/media'+path_save)
    #from os.path import basename
    #from django.core.files import File
    model.trained_model = path_save
    model.results = score
    model.status  = True
    model.save()

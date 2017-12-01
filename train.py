import glob, os
import dicom_numpy
import dicom
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
import numpy as np
import collections
from sklearn.feature_extraction import image
import scipy.misc
import natsort
from keras import layers, models

def extract_voxel_data(list_of_dicom_files):
    datasets = [dicom.read_file(f) for f in list_of_dicom_files]
    try:
        voxel_ndarray, ijk_to_xyz, slices = dicom_numpy.combine_slices(datasets)
    except dicom_numpy.DicomImportException as e:
        # invalid DICOM data
        raise
    return voxel_ndarray, slices

def generate_labels_from_xml(xml, shape_to_return, k_to_z):
    labeled = np.zeros(shape_to_return)

    tree = ET.parse(xml)
    root = tree.getroot()
    ns = {}
    ns['main'] = 'http://www.nih.gov'
    for nodule in root.findall('main:readingSession/main:unblindedReadNodule', ns):
        for node in nodule.findall('main:roi',ns):
            z = k_to_z[float(node.find('main:imageZposition',ns).text)]
            row_list = collections.defaultdict(list)
            for edge in node.findall('main:edgeMap',ns):
                x,y = int(edge[0].text), int(edge[1].text)
                labeled[x,y,z] = 1.
                row_list[x] += [y]
            # get first and last occurance of '1' in each row and fill indicies in between with all ones
            for xs in row_list.keys():
                labeled[xs,min(row_list[xs]):max(row_list[xs]),z].fill(1)
        break
    return labeled

def extract_segment(segment_id, data, y):
    row = int(segment_id / 8)
    col = int(segment_id % 8)

    return data[64*row:64*(row+1), 64*col:64*(col+1)], y[64*row:64*(row+1), 64*col:64*(col+1)]

def train_model(segment_id, patient_dicoms_dir, dir_of_snapshots=None):
    cur_model = None
    start_patient_index = 0

    if dir_of_snapshots is not None:
        prev_models = natsort.natsorted(os.listdir(os.path.join(dir_of_snapshots,'segment_'+str(segment_id))), reverse=True)
        if len(prev_models) is not 0:
            print('loading model...')
            cur_model = models.load_model(os.path.join(dir_of_snapshots,'segment_'+str(segment_id),prev_models[0]))
            start_patient_index = int(prev_models[0].split('v')[1].split('.')[0]) + 1
            print("Starting at patient: ", str(start_patient_index))
        else:
            cur_model = make_model()
    else:
        cur_model = make_model()

    for idx, sample in enumerate(patient_dicoms_dir[start_patient_index:], start=start_patient_index):
        lstFilesDCM = []
        for dirName, subdirList, fileList in os.walk(sample):
            for filename in fileList:
                if ".dcm" in filename.lower():  # check whether the file's DICOM
                    lstFilesDCM.append(os.path.join(dirName,filename))

        data, slices  = extract_voxel_data(lstFilesDCM)
        # k to z voxel mapping
        k_to_z = {k:v for v,k in enumerate(sorted(slices))}

        label_xml = glob.glob(os.path.join(sample,"*.xml"))
        y = generate_labels_from_xml(label_xml[0], data.shape, k_to_z)

        data, y = extract_segment(segment_id, data, y)
        data, y = data.T, y.T

        cur_model.train_on_batch(data.reshape(data.shape[0], data.shape[1], data.shape[2], 1),y.reshape(y.shape[0], y.shape[1], y.shape[2], 1))
        loss = cur_model.evaluate(data.reshape(data.shape[0], data.shape[1], data.shape[2], 1),y.reshape(y.shape[0], y.shape[1], y.shape[2], 1), data.shape[0])
        print(loss)
        if(idx % 50 == 0):
            print('Saving Model')
            cur_model.save(os.path.join(dir_of_snapshots,'segment_'+str(segment_id), 'model_'+str(segment_id)+'v'+str(idx)+'.h5'))

def init_model_snapshots_dir(dir_of_snapshots):
    if not os.path.exists(dir_of_snapshots):
        os.makedirs(dir_of_snapshots)
    for i in range(64):
        if not os.path.exists(os.path.join(dir_of_snapshots,'segment_'+str(i))):
            os.makedirs(os.path.join(dir_of_snapshots,'segment_'+str(i)))
    return

def make_model():
    model = models.Sequential()
    model.add(layers.Conv2D(1, (1, 1), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.Conv2D(1, (1, 1), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2DTranspose(1, (17, 17)))
    model.add(layers.Conv2DTranspose(1, (17, 17)))
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

primary = os.path.join(".","data","LIDC-IDRI")
patient_folders = [os.path.join(primary,f) for f in os.listdir(primary) if os.path.isdir(os.path.join(primary, f))]

patient_dicoms_dir = []
for sample in patient_folders:
    len_list = [len(os.listdir(os.path.join(sample,fil,os.listdir(os.path.join(sample, fil))[0]))) for fil in os.listdir(sample)]
    correct_file = os.path.join(sample, os.listdir(sample)[len_list.index(max(len_list))])
    patient_dicoms_dir.append(os.path.join(correct_file,os.listdir(correct_file)[0]))

train_model(45, patient_dicoms_dir)

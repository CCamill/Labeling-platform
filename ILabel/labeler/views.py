from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import os
# from iLabel.settings import BASE_DIR
from django.conf import settings
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import json
import tensorflow as tf
import cv2
from labeler import model
import zipfile
from io import BytesIO

BASE_DIR = settings.BASE_DIR

# Create your views here.

static_dir = os.path.join(BASE_DIR, "static")
upload_dir = os.path.join(static_dir, "upload")
endo_model_dir = os.path.join(os.path.join(BASE_DIR,"labeler"),"endo_model")
epi_model_dir = os.path.join(os.path.join(BASE_DIR,"labeler"),"epi_model")
predict_gate = [1,9]
global num_progress
global mode_index
mode_index = 1

def index(request):
    global mode_index
    global num_progress
    mode_index = 1
    num_progress = 0
    return render(request, "index.html")


def upload(request):
    if request.method == 'POST':
        dir = request.FILES

        #列表，元素为image_slice3.mat
        dirlist = dir.getlist('files')

        #列表，元素为Patient008902/Gate8/Endocardial_contour_slice1.mat
        pathlist = request.POST.getlist('paths') 

        #Patient008902
        patient_name = pathlist[0][:pathlist[0].rfind("/Gate")]
        # noinspection PyInterpreter
        if not dirlist:
            return HttpResponse('files not found')
        else:
            for file in dirlist:
                #\static\upload\Patient008902
                patient_position = os.path.join(upload_dir, patient_name)

                #\static\upload\Patient009087\Patient009087/Gate6  dirlist.index(file)返回file的下标（下标从零开始）
                position = os.path.join(patient_position, '/'.join(pathlist[dirlist.index(file)].split('/')[:-1]))
                if not os.path.exists(position):
                    os.makedirs(position)
                storage = open(position + '/' + file.name, 'wb+')
                #避免文件太大占用系统内存
                for chunk in file.chunks():
                    storage.write(chunk)
                storage.close()
            #upload\Patient008902_s\processed\prediction
            prediction_path = trans_mat_to_png(patient_position, patient_name)


            predict_patient_slices_endo(prediction_path, patient_position, patient_name)
            predict_patient_slices_epi(prediction_path, patient_position, patient_name)
            
            

            context = {
                "File_dir": json.dumps("/static/upload/"),
                "Patient_name": json.dumps(patient_name),
                "Gate_index": 1,
                "Slice_index": 1,
                "Contour_index": 1,
            }
            return JsonResponse(context)


#mat文件转png
def trans_mat_to_png(patient_position, patient_name):
    #static\upload\Patient008902\processed
    processed_dir = os.path.join(patient_position, "processed")
    #\static\upload\Patient008902\processed\prediction
    prediction_path = os.path.join(processed_dir,"prediction")

    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)
    for gate_index in range(predict_gate[0], predict_gate[1]):

        gate_path = os.path.join(processed_dir, "Gate{}".format(gate_index))
        gate_slices = []
        if not os.path.exists(gate_path):
            os.makedirs(gate_path)
        for slice_index in range(1, 33):
            # 数据已经上传到后端，从后端获取数据。
            #static\upload\Patient008902\Patient008902\Gate1
            image_slice_mat_path = os.path.join(os.path.join(os.path.join(patient_position,
                                                                          patient_name), "Gate{}".format(gate_index)),
                                                "image_slice{}.mat".format(slice_index))
            #static\upload\Patient008902\processed\Gate1
            image_slice_png_path = os.path.join(os.path.join(processed_dir, "Gate{}".format(gate_index))
                                                , "image_slice{}.png".format(slice_index))
            #加载mat文件的方式 scipy.io.loadmat()
            mat_data = scio.loadmat(image_slice_mat_path)

            #提取mat文件中的而位矩阵给 image_slice_mat_data
            image_slice_mat_data = mat_data[list(mat_data.keys())[-1]]
            #改动前 plt.imsave(os.path.join(image_slice_png_path), image_slice_mat_data)
            #将image_slice_mat_data矩阵数据覆盖到image_slice_png_path图片上
            plt.imsave(image_slice_png_path, image_slice_mat_data)
            #np.array(image_slice_mat_data)把列表转换成数组
            gate_slices.append(trans_to_predict(np.array(image_slice_mat_data)))
        prediction_gate_path = os.path.join(prediction_path,"Gate{}".format(gate_index))
        if not os.path.exists(prediction_gate_path):
            os.makedirs(prediction_gate_path)
        #np.array(gate_slices),axis=0) 给数组数组gata_slices在第一个维度地方添加维度
        #np.expand_dims( np.expand_dims(np.array(gate_slices),axis=0) , axis=-1)在最后一个维度添加添加维度
        save_npy = np.expand_dims(np.expand_dims(np.array(gate_slices),axis=0),axis=-1)
        #保存npy文件
        np.save(os.path.join(prediction_gate_path,"gate{}.npy".format(gate_index)),save_npy)
    return prediction_path

def predict_patient_slices_endo(prediction_path, patient_position, patient_name):
    print("Predicting")
    global mode_index
    global num_progress
    mode_index = 1
    num_progress = 0
    tf.reset_default_graph()
    x = tf.placeholder('float', shape=[None, 32, 32, 32, 1])
    # logits = model.create_UNet(x, 12, 2, dim=3)  # U-Net
    logits = model.create_VNet(x, 12, 2, dim=3)  # V-Net
    predicter = tf.nn.softmax(logits)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #调用模型
        saver.restore(sess, os.path.join(endo_model_dir, 'params_0'))
        for i in range(predict_gate[0],predict_gate[1]):
            mat_data = scio.loadmat(os.path.join(os.path.join(os.path.join(patient_position,
                                                patient_name), "Gate{}".format(i)),"image_slice1.mat"))
            mat_shape = mat_data[list(mat_data.keys())[-1]].shape
            test_data = np.load(os.path.join(os.path.join(prediction_path,"Gate{}".format(i)),"gate{}.npy".format(i)))
            patient_fold_prediction = []
            test_arr = test_data
            patient_fold_prediction.append(sess.run(predicter, feed_dict={x: test_arr}))
            test_arr = np.flip(test_data, axis=3 - 1)
            patient_fold_prediction.append(np.flip(sess.run(predicter, feed_dict={x: test_arr}), axis=3 - 1))
            test_arr = np.flip(test_data, axis=3)
            patient_fold_prediction.append(np.flip(sess.run(predicter, feed_dict={x: test_arr}), axis=3))
            test_arr = np.flip(np.flip(test_data, axis=3 - 1), axis=3)
            patient_fold_prediction.append(
                np.flip(np.flip(sess.run(predicter, feed_dict={x: test_arr}), axis=3 - 1), axis=3))
            final_prediction = np.squeeze(np.argmax(np.mean(np.array(patient_fold_prediction), axis=0),axis=-1))
            print("Gate{}Done".format(i))
            for q in range(0,32):
                num_progress = (i * 32 + (q + 1)) / (32 * 8)*100
                countour_image = trans_to_original_scale(final_prediction[q], mat_shape)
                np.save(os.path.join(os.path.join(prediction_path,"Gate{}".format(i)),"endo_slice{}.npy".format(q+1)),countour_image)
                # plt.imsave(os.path.join(os.path.join(prediction_path,"Gate{}".format(i)),"endo_slice{}.png".format(q+1)),countour_image)

def predict_patient_slices_epi(prediction_path, patient_position, patient_name):
    print("Predicting")
    global mode_index
    global num_progress
    mode_index = 3
    num_progress = 0
    tf.reset_default_graph()
    x = tf.placeholder('float', shape=[None, 32, 32, 32, 1])
    # logits = model.create_UNet(x, 12, 2, dim=3)  # U-Net
    logits = model.create_VNet(x, 12, 2, dim=3)  # V-Net
    predicter = tf.nn.softmax(logits)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #调用模型
        saver.restore(sess, os.path.join(epi_model_dir, 'params_0'))
        for i in range(predict_gate[0],predict_gate[1]):
            mat_data = scio.loadmat(os.path.join(os.path.join(os.path.join(patient_position,
                                                patient_name), "Gate{}".format(i)),"image_slice1.mat"))
            mat_shape = mat_data[list(mat_data.keys())[-1]].shape
            test_data = np.load(os.path.join(os.path.join(prediction_path,"Gate{}".format(i)),"gate{}.npy".format(i)))
            patient_fold_prediction = []
            test_arr = test_data
            patient_fold_prediction.append(sess.run(predicter, feed_dict={x: test_arr}))
            test_arr = np.flip(test_data, axis=3 - 1)
            patient_fold_prediction.append(np.flip(sess.run(predicter, feed_dict={x: test_arr}), axis=3 - 1))
            test_arr = np.flip(test_data, axis=3)
            patient_fold_prediction.append(np.flip(sess.run(predicter, feed_dict={x: test_arr}), axis=3))
            test_arr = np.flip(np.flip(test_data, axis=3 - 1), axis=3)
            patient_fold_prediction.append(
                np.flip(np.flip(sess.run(predicter, feed_dict={x: test_arr}), axis=3 - 1), axis=3))
            final_prediction = np.squeeze(np.argmax(np.mean(np.array(patient_fold_prediction), axis=0),axis=-1))
            print("Gate{}Done".format(i))
            for q in range(0,32):
                num_progress = (i*32+(q+1))/(32*8)*100
                countour_image = trans_to_original_scale(final_prediction[q], mat_shape)
                np.save(os.path.join(os.path.join(prediction_path,"Gate{}".format(i)),"epi_slice{}.npy".format(q+1)),countour_image)
                # plt.imsave(os.path.join(os.path.join(prediction_path, "Gate{}".format(i)), "epi_slice{}.png".format(q + 1)),countour_image)

#将图片像素标准化，少的补零，多的剪切
def trans_to_predict(img):
    #y,x分别为长和宽的像素
    y, x = img.shape
    if y < 32:
        if (32-y)%2== 0:
            a = (32-y)//2
            b = (32-y)//2
        else:
            a = (32-y)//2 + 1
            b = (32-y)//2
        img = np.pad(img, ( (a, b),(0, 0)), 'constant', constant_values=0)
        y, x = img.shape
    if x < 32:
        if (32-x)%2 == 0:
            c = (32-x)//2
            d = (32-x)//2
        else:
            c = (32-x)//2 + 1
            d = (32-x)//2
        img = np.pad(img, ((0, 0),(c, d)),  'constant', constant_values=0)
        y, x = img.shape
    starty = y // 2 - (32 // 2)
    startx = x // 2 - (32 // 2)
    trans_img = img[starty:starty + 32, startx:startx + 32]
    return trans_img

def trans_to_original_scale(data,shape):
    y, x = shape[0], shape[1]
    if x < 32 or y < 32:
        a, b, c, d = 0, 0, 0, 0
        if y < 32:
            if (32 - y) % 2 == 0:
                a = (32 - y) // 2
            else:
                a = (32 - y) // 2 + 1
        if x < 32:
            if (32 - x) % 2 == 0:
                c = (32 - x) // 2
            else:
                c = (32 - x) // 2 + 1
        data = data[a:a + y, c:c + x]
    a = y // 2 - (data.shape[0] // 2)
    b = y - a - data.shape[0]
    c = x // 2 - (data.shape[1] // 2)
    d = x - c - data.shape[1]
    trans_data = np.pad(data, ((a, b), (c, d)),  'constant', constant_values=0)
    return trans_data

#获取预测轮廓
def get_predict_contours(request):
    patient_name = eval(request.GET.getlist("patient_name")[0])
    gate_index = int(request.GET.getlist("gate_index")[0])
    slice_index = int(request.GET.getlist("slice_index")[0])

    #\static\upload\Patient008902\processed\prediction\Gate1
    prediction_dir = os.path.join(os.path.join(os.path.join(os.path.join(upload_dir, patient_name), "processed")
                                               , "prediction"),"Gate{}".format(gate_index))

    contours = [[], [], []]
    for contour_index in range(1,4):
        if contour_index == 1:
            mode = "endo"
        elif contour_index == 2:
            mode = "midmyo"
        else:
            mode = "epi"
        current_dir = os.path.join(prediction_dir, "{}_slice{}.npy".format(mode, slice_index))
        if os.path.exists(current_dir):
            contours[contour_index-1] = get_contour_from_npy(patient_name, gate_index, slice_index, contour_index)
        else:
            contours[contour_index-1] = get_contour_from_mat(patient_name, gate_index, slice_index, contour_index)
    context = {
        "contours": json.dumps(contours, cls=MyEncoder),
    }
    return JsonResponse(context)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        else:
            return super(NpEncoder, self).default(obj)

#从mat获取轮廓
def get_contour_from_mat(patient_name,gate_index,slice_index,contour_index):
    #static\upload\Patient008902\Patient008902
    dp_dir = os.path.join(os.path.join(upload_dir,patient_name),patient_name)
    #\static\upload\Patient008902\processed\prediction
    prediction_path = os.path.join(os.path.join(upload_dir,patient_name),"processed/prediction")
    if contour_index == 1:
        mode = "Endocardial_contour"
    elif contour_index == 2:
        mode = "MidMyocardial_contour"
    else:
        mode = "Epicardial_contour"
    current_countour_mat = scio.loadmat(os.path.join(os.path.join(dp_dir, "Gate{}".format(gate_index)),
                                                "{}_slice{}.mat".format(mode, slice_index)))
    current_countour_img = current_countour_mat[list(current_countour_mat.keys())[-1]]*255
    # plt.imsave(os.path.join(os.path.join(prediction_path, "Gate{}".format(gate_index)), "{}_slice{}.png".format(slice_index)),
    #            current_countour_img)
    current_countour_img = np.flipud(np.array(current_countour_img, dtype="uint8"))
    current_countour, hierarchy = cv2.findContours(current_countour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    current_countour = trans_points(current_countour)
    return current_countour

#从PNG获取轮廓
def get_contour_from_npy(patient_name,gate_index,slice_index,contour_index):
    
    prediction_dir = os.path.join(os.path.join(os.path.join(upload_dir,patient_name),"processed"),"prediction")
    if contour_index == 1:
        mode = "endo"
    elif contour_index == 2:
        mode = "midmyo"
    else:
        mode = "epi"
    current_countour_img = np.load(os.path.join(os.path.join(prediction_dir,"Gate{}".format(gate_index)),
                                                   "{}_slice{}.npy".format(mode,slice_index)))*255
    current_countour_img = np.flipud(np.array(current_countour_img,dtype="uint8"))
    current_countour, hierarchy = cv2.findContours(current_countour_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    current_countour = trans_points(current_countour)
    return current_countour

def trans_points(current_contour):
    contour = []
    for a in current_contour[0]:
        point = []
        for b in a:
            point.append(b[0])
            point.append(b[1])
        contour.append(point)
    return contour

def gen_zip_with_zipfile(start_dir,patient_name,gate_name):
    start_dir = start_dir  # 要压缩的文件夹路径

    file_news = os.path.join(os.path.join(upload_dir,patient_name),gate_name) + '.zip'  # 压缩后文件夹的名字

    z = zipfile.ZipFile(file_news, 'w', zipfile.ZIP_DEFLATED)
    for dir_path, dir_names, file_names in os.walk(start_dir):
        f_path = dir_path.replace(start_dir, '')  # 这一句很重要，不replace的话，就从根目录开始复制
        f_path = f_path and f_path + os.sep or ''  # 实现当前文件夹以及包含的所有文件的压缩
        for filename in file_names:
            z.write(os.path.join(dir_path, filename), f_path + filename)
    z.close()

def download_zipfile(request):
    patient_name = eval(request.POST.getlist("patient_name")[0])
    patient_path = os.path.join(os.path.join(upload_dir,patient_name),patient_name)
    gate_index = eval(request.POST.getlist("gate_index")[0])
    gate_name = 'Gate{}'.format(gate_index)
    gate_path = os.path.join(patient_path,gate_name)
    gen_zip_with_zipfile(gate_path,patient_name,gate_name)
    context = {
        "File_dir": json.dumps("/static/upload/"),
        "Patient_name": json.dumps(patient_name),
        "gate_dir": json.dumps(gate_name),
    }
    return JsonResponse(context)
def gen_zip_with_zipfile_patient(start_dir):
    start_dir = start_dir  # 要压缩的文件夹路径
    file_news = start_dir + '.zip'  # 压缩后文件夹的名字

    z = zipfile.ZipFile(file_news, 'w', zipfile.ZIP_DEFLATED)
    for dir_path, dir_names, file_names in os.walk(start_dir):
        f_path = dir_path.replace(start_dir, '')  # 这一句很重要，不replace的话，就从根目录开始复制
        f_path = f_path and f_path + os.sep or ''  # 实现当前文件夹以及包含的所有文件的压缩
        for filename in file_names:
            z.write(os.path.join(dir_path, filename), f_path + filename)
    z.close()
    
def download_patient_zipfile(request):
    patient_name = eval(request.POST.getlist("patient_name")[0])
    patient_path = os.path.join(os.path.join(upload_dir,patient_name),patient_name)
    gen_zip_with_zipfile_patient(patient_path)
    context = {
        "File_dir": json.dumps("/static/upload/"),
        "Patient_name": json.dumps(patient_name),
    }
    return JsonResponse(context)

def show_progress(request):
    global num_progress
    global mode_index
    if mode_index == 1:
        mode = "Endo"
    if mode_index == 2:
        mode = "MidMyo"
    if mode_index == 3:
        mode = "Epi"
    context = {
        "mode":json.dumps(mode,cls=MyEncoder),
        "numprogress": num_progress,
    }
    return JsonResponse(context, safe=False)

def save_current_image(request):
    endo_contour = eval(request.POST.getlist("endo_contour")[0])
    midmyo_contour = eval(request.POST.getlist("midmyo_contour")[0])
    epi_contour = eval(request.POST.getlist("epi_contour")[0])
    patient_name = eval(request.POST.getlist("patient_name")[0])
    gate_index = int(request.POST.getlist("gate_index")[0])
    slice_index = int(request.POST.getlist("slice_index")[0])
    for contour_index in range(1,4):
        if contour_index == 1:
            mode = ["Endocardial_contour","endo"]
            contour = endo_contour
        elif contour_index == 2:
            mode = ["MidMyocardial_contour","midmyo"]
            contour = midmyo_contour
        else:
            mode = ["Epicardial_contour","epi"]
            contour = epi_contour
        update_original_mat(patient_name,gate_index,slice_index,mode,contour)
    return JsonResponse(1,safe=False)

def update_original_mat(patient_name,gate_index,slice_index,mode,contour):
    #\static\upload\Patient008902\Patient008902
    dp_dir = os.path.join(os.path.join(upload_dir,patient_name),patient_name)

    current_countour_mat = scio.loadmat(os.path.join(os.path.join(dp_dir, "Gate{}".format(gate_index)),
                                                     "image_slice{}.mat".format(slice_index)))
    prediction_dir = os.path.join(os.path.join(os.path.join(upload_dir, patient_name), "processed"), "prediction")

    rename_path = os.path.join(os.path.join(prediction_dir,"Gate{}".format(gate_index)),
                                                   "{}_slice{}.npy".format(mode[1],slice_index))

    newname_path = os.path.join(os.path.join(prediction_dir,"Gate{}".format(gate_index)),
                                                   "{}_slice{}.npy".format(mode[0],slice_index))
    #获取数组维度给mat_shape
    mat_shape = current_countour_mat[list(current_countour_mat.keys())[-1]].shape

    key = list(current_countour_mat.keys())[-1]
    #获得一个维度为mat_shape且全为零的矩阵
    mat_origin = np.zeros(mat_shape)
    res = []
    for point_index in range(len(contour)):
        if point_index != len(contour) - 1:
            res.append(np.array(getLinePoint(contour[point_index], contour[point_index + 1])))
        else:
            res.append(np.array(getLinePoint(contour[point_index], contour[0])))
    for row in res:
        for point in row:
            mat_origin[point[1]][point[0]] = 1
    mat_origin = np.flipud(mat_origin)
    scio.savemat(os.path.join(os.path.join(dp_dir, "Gate{}".format(gate_index)),
                "{}_slice{}.mat".format(mode[0], slice_index)),{key:mat_origin})
    if os.path.exists(rename_path):
        os.rename(rename_path, newname_path)
    plt.imsave(os.path.join(os.path.join(dp_dir, "Gate{}".format(gate_index)),
                "{}_slice{}.png".format(mode[0], slice_index)),mat_origin)

def getLinePoint(A, B):
    x0 = A[0]
    x1 = B[0]
    y0 = A[1]
    y1 = B[1]
    if x1 - x0 == 0:
        pos = np.array(list(zip([x0] * 256, np.linspace(y0, y1, 256) + 0.5)))
        pos = pos.astype(np.int32)
        return np.unique(pos, axis=0)
    else:
        k = (y1 - y0) / (x1 - x0)

        def func(x):
            return k * (x - x0) + y0

        x_v = np.linspace(x0, x1, 256)
        pos = np.array(list(zip(x_v, func(x_v) + 0.5)))
        pos = pos.astype(np.int32)
        return np.unique(pos, axis=0)
        
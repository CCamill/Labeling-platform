def get_contour_from_png(contours, patient_name,gate_index,slice_index,contour_index):
    prediction_dir = os.path.join(os.path.join(os.path.join(upload_dir,patient_name),"processed"),"prediction")
    if contour_index == 1:
        mode = "endo"
    elif contour_index == 2:
        mode = "middle"
    else:
        mode = "epi"
    current_countour_img = np.load(os.path.join(os.path.join(prediction_dir,"Gate{}".format(gate_index)),
                                                   "{}_slice{}.npy".format(mode,slice_index)))*255
    current_countour_img = np.array(current_countour_img,dtype="uint8")
    current_countour, hierarchy = cv2.findContours(current_countour_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    current_countour = trans_points(current_countour)
    contours[contour_index] = current_countour
    return contours

def trans_points(current_contour):
    contour = []
    for a in current_contour[0]:
        point = []
        for b in a:
            point.append(b[0])
            point.append(b[1])
        contour.append(point)
    return contour



def generate_dataset(input_folder, output_folder):
    assert os.path.isdir(input_folder) and os.path.isdir(output_folder)
    output_folder_1 = os.path.join(output_folder, "1")
    if not os.path.exists(output_folder_1):
        os.mkdir(output_folder_1)

    output_folder_0 = os.path.join(output_folder, "0")
    if not os.path.exists(output_folder_0):
        os.mkdir(output_folder_0)

    video_files = [os.path.join(input_folder, item) for item in os.listdir(input_folder) if item.endswith('.avi')]
    # video_files = [os.path.join(input_folder, item) for item in os.listdir(input_folder) if
    #                item == '2020-01-18_(07-04-58)_002FBE71E101_85136_12950.avi']

    print(video_files)

    for video_file in tqdm(video_files):
        video_stream = cv2.VideoCapture(video_file)
        grab, frame = video_stream.read()

        cnt = 0

        while frame is not None:
            if cnt % 60 == 0:
                showframe = deepcopy(frame)
                # text coords were 40, 40
                showframe = cv2.putText(showframe, "1: Move,2: Idle,3: Discard", (40, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
                cv2.imshow('frame', showframe)
                img_path = os.path.basename(video_file).replace(".avi", "") + "%d.jpg" % video_stream.get(
                    cv2.CAP_PROP_POS_FRAMES)
                frame = cv2.resize(frame, (224, 224))
                if cv2.waitKey(0) == 49:
                    img_path = os.path.join(output_folder_1, img_path)
                    frame = cv2.resize(frame, (224, 224))
                    cv2.imwrite(img_path, frame)
                elif cv2.waitKey(0) == 50:
                    img_path = os.path.join(output_folder_0, img_path)
                    frame = cv2.resize(frame, (224, 224))
                    cv2.imwrite(img_path, frame)

            grab, frame = video_stream.read()
            cnt += 15
            # if frame is not None:
            #     print('has frame')
    
    time.sleep(3)
    video_stream.release()


def prepare_for_training(output_folder):
    pos_folder = os.path.join(output_folder, '1')
    neg_folder = os.path.join(output_folder, '0')
    assert os.path.exists(pos_folder) and os.path.exists(neg_folder)

    x = []
    y = []

    for img_path in os.listdir(pos_folder):
        img_path = os.path.join(pos_folder, img_path)
        x.append(cv2.imread(img_path))
        y.append(1)
    for img_path in os.listdir(neg_folder):
        img_path = os.path.join(neg_folder, img_path)
        x.append(cv2.imread(img_path))
        y.append(0)

    x = np.asarray(x)
    y = np.asarray(y)

    index = np.random.permutation(x.shape[0])
    x = x[index]
    y = y[index]
    return x, y

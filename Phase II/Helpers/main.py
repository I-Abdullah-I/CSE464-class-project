src_path = 'test2.mp4'
result_path = 'test_2_out.avi'
# result_path = 'detection_results_scale_only_1.5_y_scanned_256_modulo_5_semi_final_V.1.avi'

# openCV's configurations to save the video on disk.
vid = cv2.VideoCapture(src_path)
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(result_path, codec, fps, (width, height))

accepted_frame = 0
frame_count = 0
bboxes = []
while True:
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if accepted_frame == 0:
            frame, bboxes = process_image(frame)
            frame = cv2.putText(frame, 'Obtained bbox', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif bboxes:
            frame = cv2.putText(frame, 'Interpolated bbox', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            for bbox in bboxes:
                frame = cv2.rectangle(frame, bbox[0], bbox[1], (0,255,0), 6)
        accepted_frame = (accepted_frame + 1) % 5
        frame_count += 1
        print("Frame number: ", frame_count)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # if frame_count > 100:
        #     cv2.imshow('frame no.{}'.format(frame_count), frame)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # elif frame_count == 1:
        #     break
        out.write(frame)
        
    else:
        print('Video ended or an error has occurred.')
        break

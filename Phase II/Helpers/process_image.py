def process_image(img):
    heatmaps = collections.deque(maxlen=29)
    # global heatmaps

    heat = np.zeros_like(img[:,:,0]).astype(np.float32)
    
    # ystarts = [400,350,350]
    # ystops = [656,570,570]
    ystarts = [400]
    ystops = [656]
    # ystarts = [200,175,175]
    # ystops = [328,285,285]
    
    # Look for cars at different scales
    # scales = [1., 1.5, 2.0]
    # scales = [1.0, 2.0]
    scales = [1.5]
    for scale, ystart, ystop  in zip(scales, ystarts, ystops):
        box_list, out_img, out_img_windows  = find_cars(img, ystart, ystop, scale, svc_l, X_scaler_l, orient_l, pix_per_cell_l, cell_per_block_l, spatial_size_l, hist_bins_l)
        heat = add_heat(heat,box_list)
    # Append heatmap and compute the sum of the last n ones
    heatmaps.append(heat)
    sum_heatmap = np.array(heatmaps).sum(axis=0)
    # Apply the threshold to remove false positives
    # heat = apply_threshold(sum_heatmap, min(len(heatmaps) * 1, 28))
    print(min(len(heatmaps) * 1, 28))
    heat = apply_threshold(sum_heatmap, 0)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img, bbox = draw_labeled_bboxes(np.copy(img), labels)
    
    # visualizing heatmaps in numbers
    # print('threshold = {}'.format(min(len(heatmaps) * 1, 28)))
    # cv2.imshow('sum_heatmap: ', sum_heatmap)
    # cv2.imshow('thresholded heatmap: ', heat)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return draw_img, bbox
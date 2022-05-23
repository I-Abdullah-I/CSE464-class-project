# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    draw_img_all_windows = np.copy(img)
    img = img.astype(np.float32)/255

    
    bbox_list = []
    
    img_tosearch = img[ystart:ystop,:,:]
    ####################################
    # img_tosearch_temp = cv2.putText(np.copy(img_tosearch), 'img size = {}'.format(img_tosearch.shape), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # cv2.imshow('img_tosearch_temp', img_tosearch_temp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ####################################
    ctrans_tosearch =  cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int32(imshape[1]/scale), np.int32(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            #test_prediction = svc.predict(test_features)
            test_prediction = svc.decision_function(test_features)
 
            xbox_left = np.int32(xleft*scale)
            ytop_draw = np.int32(ytop*scale)
            win_draw = np.int32(window*scale)
            cv2.rectangle(draw_img_all_windows,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
            # cv2.imshow('draw_img_all_windows', draw_img_all_windows)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            #if test_prediction == 1:
            if test_prediction > 0.4:
                xbox_left = np.int32(xleft*scale)
                ytop_draw = np.int32(ytop*scale)
                win_draw = np.int32(window*scale)
                bbox_list.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
                # str_on_img_1 = 'xbox_left: {} - ytop_draw+ystart: {} - xbox_left+win_draw: {}'.format(xbox_left, ytop_draw+ystart, xbox_left+win_draw)
                # str_on_img_2 = 'ytop_draw+win_draw+ystart: {}'.format(ytop_draw+win_draw+ystart)
                # img = cv2.putText(draw_img, str_on_img_1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                # img = cv2.putText(img, str_on_img_2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
    return bbox_list, draw_img, draw_img_all_windows
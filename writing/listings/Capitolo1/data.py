import nuympy as np

# ...

for i in range(n_soggetti):
    if n_soggetti < 2:
        LOG("Loading data and labels")
        
        multiple_subj = False

        X = data[0].values.reshape(n_samples, n_points, n_series)
        y = labels[0].values

        test_idx_end = int((n_samples // 2) + int(n_samples * 0.1))
        test_idx_begin = int((n_samples // 2) - int(n_samples * 0.1))

        # Create a boolean mask to select elements to keep for training and testing sets
        mask_train = np.ones(X.shape[0], dtype=bool)
        mask_train[test_idx_begin:test_idx_end] = False
        X_train = X[mask_train, :, :]
        y_train = y[mask_train]

        mask_test = np.zeros(X.shape[0], dtype=bool)
        mask_test[test_idx_begin:test_idx_end] = True
        X_test = X[mask_test, :, :]
        y_test = y[mask_test]

        X_train_2D = X_train.reshape(X_train.shape[0], -1)
        X_test_2D = X_test.reshape(X_test.shape[0], -1)

        # Apply StandardScaler
        X_train = scaler.fit_transform(X_train_2D)
        X_test = scaler.transform(X_test_2D)

        # Reshape back to 3D for CNN
        X_train = X_train.reshape(X_train.shape[0], n_points, n_series)
        X_test = X_test.reshape(X_test.shape[0], n_points, n_series)

        LOG("Data and labels loaded!")
        LOG(f'Train data: {X_train.shape}, test data: {X_test.shape}, train labels: {y_train.shape} test labels: {y_test.shape}')
    else:
        #---------------------------DATA GENERATION-------------------------------#
        if n_soggetti > 1:
            n_samples_overall = 0
            X_train = []
            X_test = []
            LOG(f"Loading data of subject {i+1}...")
            for j in range(n_soggetti):
                # Dataset di test
                if j == i:
                    X_test = data[i].values
                    X_test = scaler.fit_transform(X_test)
                    X_test = X_test.reshape(n_samples, n_points, n_series)
                    continue

                # Dataset di training
                X_train.extend(data[j].values)  # append to the whole time series
                n_samples_overall += int(data_files[j].name[-11:-7])

            X_train = scaler.fit_transform(X_train)
            X_train = X_train.reshape(n_samples_overall, n_points, n_series)
        
        LOG("Data loaded!")
        LOG(f'Train data: {X_train.shape}')
        LOG(f'Test data: {X_test.shape}')

        #---------------------------LABEL GENERATION-------------------------------#

        y_train = []
        y_test = []
        first_iter = True
        LOG(f"Loading labels of subject {i+1}...")

        for j in range(n_soggetti):
            # Label di testing
            if j == i:
                y_test = labels[i].values
                continue
            # Label di training
            if first_iter:
                y_train = labels[j].values
                first_iter = False
            else:
                y_train = np.append(y_train, labels[j].values, axis=0)
    
        LOG("Labels loaded!")
        LOG(f'Training labels: {y_train.shape}')
        LOG(f'Testing labels: {y_test.shape}')
# ...

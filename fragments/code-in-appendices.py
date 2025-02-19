
# Image Augmentation

img = cv2.imread(img_path)
data = img_to_array(img)

# expand dimension
samples = expand_dims(data, 0)

# create image data augmentation generator
datagen = ImageDataGenerator(zoom_range=[0.8,1.2], horizontal_flip=True, vertical_flip=True)

#prepare the iterator
it = datagen.flow(samples, batch_size=1)
batch = it.next()

# convert to unsigned integers
image = batch[0].astype('uint8')

# write augmented image to disc
cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


## ---------------------

# Split training and test set, normalise and PCA for SVC

# split into train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# normalise features
input_std = StandardScaler()
input_std.fit(Xtrain)
Xtrain_std = input_std.transform(Xtrain)
Xtest_std = input_std.transform(Xtest)

# do the PCA, choose the numbher of components to retain
input_pca = PCA(n_components=100)
input_pca.fit(Xtrain_std)
Xtrain_std_pca = input_pca.transform(Xtrain_std)
Xtest_std_pca = input_pca.transform(Xtest_std)

C = 10
model = SVC(kernel='rbf', C=C)

# train the model
model.fit(Xtrain_std_pca, ytrain)
y_pred_train = model.predict(Xtrain_std_pca)

# predict on test set
y_pred = model.predict(Xtest_std_pca)

## ---------------------

# Gridsearch for SVM

std = StandardScaler()
pca = PCA(n_components=50)
svc = SVC(kernel='rbf')
pipe_svc = Pipeline([('std',std),('pca', pca),('svc',svc)])

# parameters of pipelines
param_grid_svc = {
    'pca__n_components': [10, 50, 100],
    'svc__kernel': ['rbf', 'linear', 'sigmoid'],
    'svc__C': [0.1, 1, 10],
}

search_svc = GridSearchCV(pipe_svc, param_grid_svc,
                      scoring="accuracy",
                      cv=5, # default to stratified
                      verbose=3, 
                      n_jobs=3
                         )

# perform grid-search
%time search_svc.fit(Xtrain, ytrain)

print("Best parameter (CV score=%0.3f):" % search_svc.best_score_)
print(search_svc.best_params_)



## ---------------------

# Split training and test set, normalise and PCA for MLP

# split into train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,test_size=0.3, random_state=42, stratify=y)

# normalise features
input_std = StandardScaler()
input_std.fit(Xtrain)
Xtrain_std = input_std.transform(Xtrain)
Xtest_std = input_std.transform(Xtest)

# do the PCA, choose the numbher of components to retain
input_pca = PCA(n_components=100)
input_pca.fit(Xtrain_std)
Xtrain_std_pca = input_pca.transform(Xtrain_std)
Xtest_std_pca = input_pca.transform(Xtest_std)

# model initialization
hidden_layer_size = 500
max_iter = 1000
mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_size), max_iter=max_iter, alpha=0.0001,
                    solver='sgd', verbose=0, tol=0.000001,
                    early_stopping=False, momentum=0.9)

# train the Model
h = mlp.fit(Xtrain_std_pca, ytrain)

# predict on test set
y_pred = mlp.predict(Xtest_std_pca)



## ---------------------

# Grid search for MLP

std = StandardScaler()
pca = PCA(n_components=100)
mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_size), max_iter=max_iter, alpha=0.001,
                    solver='sgd', verbose=0, tol=0.000001,
                    early_stopping=False, momentum=0.9)
pipe_mlp = Pipeline([('std',std),('pca', pca),('mlp',mlp)])

# parameters of pipelines
param_grid_mlp = {
    'pca__n_components': [10, 50, 100],
    'mlp__solver': ['sgd', 'adam'],
    'mlp__max_iter': [500, 1000],
    'mlp__hidden_layer_sizes': [(100), (250), (500)],
}


search_mlp = GridSearchCV(pipe_mlp, param_grid_mlp,
                      scoring="accuracy",
                      cv=5,
                      verbose=3, 
                      n_jobs=3,
                         )

# perform grid-search
%time search_mlp.fit(Xtrain, ytrain)

print("Best parameter (CV score=%0.3f):" % search_mlp.best_score_)
print(search_mlp.best_params_)


## ---------------------

# CNN model

def build_cnn_model(X, y):
    shape = (WIDTH, HEIGHT, 3)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), padding = 'same',activation = 'relu', input_shape = shape))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, kernel_size = (3,3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(128, kernel_size = (3,3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128,activation = 'relu'))
    model.add(Dense(64,activation = 'relu'))
    model.add(Dense(32,activation = 'relu'))
    model.add(Dense(3,activation = 'softmax'))

    opt = SGD(lr=0.01)
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

## ---------------------

# Train CNN Model

# split into train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# normalise features
input_std = StandardScaler()
input_std.fit(Xtrain)
Xtrain_std = input_std.transform(Xtrain)
Xtest_std = input_std.transform(Xtest)

# reshape flatten input to 3-dimension
Xtrain_std = Xtrain_std.reshape((-1, WIDTH, HEIGHT, 3))
Xtest_std = Xtest_std.reshape((-1, WIDTH, HEIGHT, 3))

# build model
model = build_cnn_model(Xtrain_std, ytrain)

# train model
no_epochs = 50
print('Training with for {0} epochs'.format(no_epochs))
history = model.fit(Xtrain_std, ytrain, validation_split = 0.2, epochs=no_epochs, verbose=1)


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

# calculate final loss on train set
loss_final = np.sqrt(float(hist['loss'].tail(1)))
print('Final Loss on training set: {}'.format(round(loss_final, 3)))


## ---------------------

# Cut images from video

PERCENT_SKIP_HEAD = 18
PERCENT_SKIP_TAIL = 12
CAPTURING_INTERVAL = 10
DEFAULT_FPS = 30

# read video from pathIn, save image to pathOut
def extract_frames(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)

    success = True
    videoDuration = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(vidcap.get(cv2.CAP_PROP_FPS))

    # calculated frames skip position and capturing interval
    skipHead = int((PERCENT_SKIP_HEAD * videoDuration * DEFAULT_FPS)/100)
    skipTail = int(((100 - PERCENT_SKIP_TAIL) * videoDuration * DEFAULT_FPS)/100)
    capturingInterval = int((CAPTURING_INTERVAL * videoDuration * DEFAULT_FPS)/100)

    imgNum = 0
    while success:
        success,image = vidcap.read()
        if count > skipHead and count < skipTail and count >= (skipHead + (capturingInterval * imgNum)):
            cv2.imwrite( pathOut + "__%d.jpg" % imgNum, image)     # save frame as JPEG file
            imgNum += 1
        count = count + 1
    vidcap.release()


## ---------------------

# CNN model for image filtering

def build_cnn_model(X, y):
    shape = (WIDTH, HEIGHT, 3)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), padding = 'same',activation = 'relu', input_shape = shape))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, kernel_size = (3,3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(64,activation = 'relu'))
    model.add(Dense(16,activation = 'relu'))
    model.add(Dense(2,activation = 'softmax'))

    opt = SGD(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

## ---------------------

# Capturing ROI (squared region)

def crop_img_round(file_name, img):
	# convert to gray
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # binary thresholding
    ret, thresh = cv2.threshold(gray_img, 64, 255, 0)

    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("find no contours for file: " + file_name)
        return None

    # bounding box around max-contour
    max_contour = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(max_contour)

    # crop square box inside bounding box
    mval = min(w, h)
    newy = math.floor(y + h/2 - mval/2)
    newx = math.floor(x + w/2 - mval/2)
    return img[newy:newy+mval, newx:newx+mval]

## ---------------------

# Merge otoscopy image with tympanometry data in 2 models

# X_ALL contains otoscopy image, stacking with tympanometry label as the last feature

# split into train and test
Xtrain_all, Xtest_all, ytrain, ytest = train_test_split(X_ALL, y, test_size=0.3, random_state=42, stratify=y)

Xtrain = Xtrain_all[:, :-1] 		# train otoscopy data
Xtrain_tymp = Xtrain_all[:, -1]		# train tympanometry data

Xtest = Xtest_all[:, :-1]			# test otoscopy data
Xtest_tymp = Xtest_all[:, -1]		# test tympanometry data


# normalise features
input_std = StandardScaler()
input_std.fit(Xtrain)
Xtrain_std = input_std.transform(Xtrain)
Xtest_std = input_std.transform(Xtest)

# do the PCA, choose the numbher of components to retain
input_pca = PCA(n_components=100)
input_pca.fit(Xtrain_std)
Xtrain_std_pca = input_pca.transform(Xtrain_std)
Xtest_std_pca = input_pca.transform(Xtest_std)

# train model with otoscopy data
C = 10
model = SVC(kernel='rbf', C=C, probability=True)
model.fit(Xtrain_std_pca, ytrain)

# predict on train set
y_pred_train = model.predict_proba(Xtrain_std_pca)

# also predict on test set (used for validation)
y_pred = model.predict_proba(Xtest_std_pca)

# merge proba output with tymp data, feed to mlp
y_pred_train = y_pred_train[:,-1]
y_pred = y_pred[:,-1]

X_train_new = np.column_stack((y_pred_train.reshape(-1,1), Xtrain_tymp.reshape(-1,1)))
X_test_new = np.column_stack((y_pred.reshape(-1,1), Xtest_tymp.reshape(-1,1)))


hidden_layer_size = 10
max_iter = 50
mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_size), max_iter=max_iter, alpha=0.0001,
                    solver='sgd', verbose=0, tol=0.000001,
                    early_stopping=False, momentum=0.9)

# train the Model
h = mlp.fit(X_train_new, ytrain)

# validate on test set
y_pred_new = mlp.predict(X_test_new)


## ---------------------
let model = null;
tf.loadLayersModel("http://localhost:5500/saved_model/model.json").then(m => {
    console.log("Model loaded");
    // m.compile();
    model = m;
})

function openCvReady() {

    cv.onRuntimeInitialized = function() {
        let video = document.getElementById("cam_input"); // video is the id of video tag
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(function(stream) {
            video.srcObject = stream;
            video.play();
        })
        .catch(function(err) {
            console.log("An error occurred! " + err);
        });
        let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
        let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
        let gray = new cv.Mat();
        let cap = new cv.VideoCapture(cam_input);
        let faces = new cv.RectVector();
        let classifier = new cv.CascadeClassifier();
        let utils = new Utils('errorMessage');
        let faceCascadeFile = 'haarcascade_frontalface_default.xml'; // path to xml
        utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
            classifier.load(faceCascadeFile); // in the callback, load the cascade from file 
        });
        const FPS = 24;
        tf.tidy(() => {

            function processVideo() {
                let begin = Date.now();
                cap.read(src);
                src.copyTo(dst);
                cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
                try{
                    classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
                    console.log(faces.size());
                }catch(err){
                    console.log(err);
                }
                for (let i = 0; i < faces.size(); ++i) {
                    let face = faces.get(i);
                    let point1 = new cv.Point(face.x, face.y);
                    let point2 = new cv.Point(face.x + face.width, face.y + face.height);
                    let img = tf.browser.fromPixels(video, 1);
                    cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
                    img = tf.slice(img, [face.y, face.x], [face.height, face.width]);
                    // console.log(img.shape);
                    if(model){
                        img = tf.image.resizeBilinear(img, [194, 194]);
                        let outputs = model.predict(tf.expandDims(img)).as2D(68, 2);
                        console.log(outputs);
                        outputs.print()
                        
                    }
                    
                }
                cv.imshow("canvas_output", dst);
                // schedule next one.
                let delay = 1000/FPS - (Date.now() - begin);
                setTimeout(processVideo, delay);
            }
            // schedule first one.
            setTimeout(processVideo, 0);
        })
    };
}
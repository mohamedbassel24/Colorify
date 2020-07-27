let express = require('express')
let router = express.Router()
const exec = require('child_process').exec;
const multer = require('multer')

const fs = require('fs');

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'model/Input_and_Output/')
    },
    filename: (req, file, cb) => {
        let re = /(?:\.([^.]+))?$/;

        let ext = re.exec(file.originalname)[1]; // "txt"

        // console.log('works1')
        cb(null, 'black.' + ext)

        // console.log('works2')
    }
});

const upload = multer({ storage: storage });

router.post('/', upload.single('image'), (req, res, next) => {
    // const contents = fs.readFileSync('model/Input_and_Output/3.mp4' , {
    //     encoding: 'base64'
    // });
    // res.send(contents)
    // console.log('works3')
    // res.send('1')

    // console.log('works4')
    // const contents = fs.readFileSync('public/black.jpg', { encoding: 'base64' });
    // console.log('works5')

    let re = /(?:\.([^.]+))?$/;
    let ext = re.exec(req.body.name)[1];

    let type = 0
    let model = 1

    type = req.body.type
    model = req.body.model 
    const cmd = 'cd model && rm stream/* && python3 main.py '+ type +' ' + model + ' black.' + ext 
    // const cmd = 'ls'
    console.log(cmd)
    exec(cmd, (err, stdout, stderr) => {
        if (err) {
            // node couldn't execute the command
            console.log(stderr)
            console.log(err)
            return;
        }
        let contents = 0
        if(type == 1)
        {
            
                contents = 'http://localhost:9000/stream/LastOutput.mp4'
          
        }
        else{
            contents = fs.readFileSync('model/Input_and_Output/ColorizedImage.png' , {
                encoding: 'base64'
            });
        }


        exec('rm model/Input_and_Output/*', (err, stdout, stderr) => {

            res.send(contents)
        })
    });

})

router.get('/', (req, res, next) => {
    // const contents = fs.readFileSync('images/black.jpg', { encoding: 'base64' });
    res.send('contents 222222')
})


module.exports = router
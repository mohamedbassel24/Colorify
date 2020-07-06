import React, {Component} from 'react';
import './upload.css'
import axios from 'axios'

class Upload extends Component{

    constructor(props) {
        super(props)
        this.state = {
            file: null,
            file2:null,
            color:null,
            black:null,
            uploadIcon:require('./images/upload2.png')
        }
        this.handleChange = this.handleChange.bind(this)
        this.Colorize = this.Colorize.bind(this)
        this.Cancel = this.Cancel.bind(this)
      }

    handleChange(e){
        if(e.target.files[0] == null)
            return
         this.setState({
            file: URL.createObjectURL(e.target.files[0]),
            file2:e.target.files[0]
        })

        let radios = document.getElementsByName('type');
        let type = 1
        for (let i = 0, length = radios.length; i < length; i++) {
            if (radios[i].checked) {
                type =  radios[i].value
                break;
            }
        }
        if(type == 0 )
        {
            let ImagePreview = document.getElementById('image-preview')
            ImagePreview.style.display= 'block'
        }
        else{
            let VideoPreview = document.getElementById('video-preview')
            VideoPreview    .style.display= 'block'
        }
        
        let ColorizeButton = document.getElementById('colorize-button')
        ColorizeButton.style.display= 'inline'
        let CancelButton = document.getElementById('cancel-button')
        CancelButton.style.display= 'inline'
        // ImagePreview.setAttribute('src',)
    }
    Cancel(e)
    {
       
        let UploadContent = document.getElementById('upload-content')
        UploadContent.style.display = 'inline'   

        let InputImage = document.getElementById('upload')
        InputImage.disabled = false

        this.setState({
            file:null
        })

        let ImagePreview = document.getElementById('image-preview')
        ImagePreview.style.display= 'none'
        let Result= document.getElementById('black-color-images')
        Result.style.display = 'none'
        let TryButton = document.getElementById('try-button')
        TryButton.style.display= 'none'
    }
    Colorize(e){
        let UploadContent = document.getElementById('upload-content')
        UploadContent.style.display = 'none'
        let Loading = document.getElementById('loading-icon')
        Loading.style.display = 'inline-block'    
        let InputImage = document.getElementById('upload')
        InputImage.disabled = true

        let that = this

        let OriginalImage = document.getElementById('image-preview').src

        let ColorizeButton = document.getElementById('colorize-button')
        ColorizeButton.style.display= 'none'
        let CancelButton = document.getElementById('cancel-button')
        CancelButton.style.display= 'none'

        let ImageToSend = this.state.file2
        let formdata = new FormData()
        // console.log(ImageToSend.name)
        formdata.append('image',ImageToSend)
        formdata.append('name',ImageToSend.name)
        let radios = document.getElementsByName('model');
        let model = 1
        for (let i = 0, length = radios.length; i < length; i++) {
            if (radios[i].checked) {
                model =  radios[i].value
                break;
            }
        }
        formdata.append('model',model)
        
        radios = document.getElementsByName('type');
        let type = 1
        for (let i = 0, length = radios.length; i < length; i++) {
            if (radios[i].checked) {
                type =  radios[i].value
                break;
            }
        }
        formdata.append('type',type)
        // const DataToSend = {
        //     name:'test',
        //     x:'y',
        //     image:formdata
        // }
        // axios({
        //         url:'http://localhost:9000/colorize',
        //         method: 'POST',
        //         data:JSON.stringify(DataToSend)
        //     })
        axios.post('/colorize',formdata)
        .then((res)=>{
            console.log(res.data)
            Loading.style.display = 'none'
            that.setState({
                color: res.data,
                black: OriginalImage
            }) 

            if (type == 0)
            {
                let Result= document.getElementById('black-color-images')
                Result.style.display = 'inline'
                
            }
            else{
                let Result= document.getElementById('black-color-videos')
                Result.style.display = 'inline'
            
            }

            let TryButton = document.getElementById('try-button')
            TryButton.style.display= 'inline'
            

        })
        .catch((err)=>{
            console.log(err)
        })


    }
    render(){
        return (
            <div  className = 'upload-main'>
            <h1>Try it Now!</h1>
            <hr></hr>
                <h2 className="upload-title">Upload your photo/video and colorize it</h2>
                <label id = 'upload-label' for='upload'>
                    <form action = '#'>
                        <input type="file" id='upload' onChange={this.handleChange} style={{display:'none'}} />
                    </form>
                        <div onDragOver=''className="upload-box">
                            <div>
                                <div id='loading-icon' class="spinner spinner--steps icon-spinner" aria-hidden="true"></div>
                            </div>
                            <div id = "upload-content">
                                <img src={this.state.file} style={{display:'none'}} id="image-preview"/>

                                <video width="320" height="240" style={{display:'none'}} id="video-preview" controls>
                                    <source src={this.state.file} type="video/mp4"  id="video-preview-src" />
                                </video>
                                
                                <div className="model">
                                    {/* <p>You want to colorize:</p> */}
                                    <input type="radio" id="human" name="model" value="0" />
                                    <label for="human">Human</label>
                                    <input type="radio" id="nature" name="model" value="1"/>
                                    <label for="nature">Nature</label>
                                </div>
                                <div className="type">
                                    {/* <p>You want to colorize:</p> */}
                                    <input type="radio" id="image" name="type" value="0" />
                                    <label for="image">Image</label>
                                    <input type="radio" id="video" name="type" value="1"/>
                                    <label for="video">Video  </label>
                                </div>
                                <img className="upload-icon"src={this.state.uploadIcon}/>
                                <p>select or drag photo</p>
                                <span className='select-photo'>Select photo/video</span>
                            </div>
                            <div id = 'black-color-images' style={{display:'none'}}>
                                {/* <img src={this.state.black}  id="image-black"/> */}
                                <img src={`data:image/png;base64,${this.state.color}`}  id="image-color"/>

                            </div>
                            <div id = 'black-color-videos' style={{display:'none'}}>
                                {/* <img src={this.state.black}  id="image-black"/> */}
                                  <embed type="video/mp4" src={`data:video/mp4;base64,${this.state.color}`}  />
    
                            </div>
                        </div>
                        
                        

                </label>
                <button className='select-photo c-button' onClick={this.Cancel} id ="cancel-button">Cancel</button>
                <button className='select-photo c-button' onClick={this.Colorize} id ="colorize-button">Colorize</button>
                <button className='select-photo c-button' onClick={this.Cancel} id ="try-button">Try Again</button>
            </div>
        )
    }
}

export default Upload
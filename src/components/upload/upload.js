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

        let ImagePreview = document.getElementById('image-preview')
        ImagePreview.style.display= 'block'
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

        formdata.append('image',ImageToSend)

        axios({
                url:'/colorize',
                method: 'POST',
                data:formdata
            })
        .then((res)=>{
            // console.log(res)
            Loading.style.display = 'none'
            that.setState({
                color: OriginalImage,
                black: OriginalImage
            }) 
            let Result= document.getElementById('black-color-images')
            Result.style.display = 'inline'
            
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
                    <input type="file" id='upload' onChange={this.handleChange} style={{display:'none'}} />
                        
                        <div onDragOver=''className="upload-box">
                            <div>
                                <div id='loading-icon' class="spinner spinner--steps icon-spinner" aria-hidden="true"></div>
                            </div>
                            <div id = "upload-content">
                                <img src={this.state.file} style={{display:'none'}} id="image-preview"/>
                                <img className="upload-icon"src={this.state.uploadIcon}/>
                                <p>select or drag photo</p>
                                <span className='select-photo'>Select photo/video</span>
                            </div>
                            <div id = 'black-color-images' style={{display:'none'}}>
                                {/* <img src={this.state.black}  id="image-black"/> */}
                                <img src={this.state.color}  id="image-color"/>

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
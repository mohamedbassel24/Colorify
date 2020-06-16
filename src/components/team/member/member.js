import React, {Component} from 'react';
import './member.css'
class Member extends Component{

    constructor(props) {
        super(props)
        this.state = {
            image: props.Image,
            FullName: props.FullName
        }

      }

    render(){
        return (


            <div className="member-image">
                <img className = "profile-pic" src={this.state.image}/>
                <p>{this.state.FullName}</p>
            </div>


        )
    }
}

export default Member
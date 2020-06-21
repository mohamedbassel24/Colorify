import React, {Component} from 'react';
import './team.css'
import Member from './member/member'
class Team extends Component{

    constructor(props) {
        super(props)

      }

    render(){
        return (
            <div  className = 'team-main'>               
                <h1>Our Team</h1>
                <hr></hr>
                <div className="team-box">
                    <Member Image ={require('./images/member2.jpg')} FullName="Mohamed Bassel" />
                    <Member Image ={require('./images/member3.jpg')} FullName="Mohamed Haitham" />
                    <Member Image ={require('./images/member11.jpg')} FullName="Mohamed Ezzat" />
                    <Member Image ={require('./images/member4.jpg')} FullName="Marwan Medhat" />
                 </div>

            </div>
        )
    }
}

export default Team
import React, {Component} from 'react';
import './header.css'
class Header extends Component{

    constructor(props) {
        super(props)
        this.state = {
          count: 0,
          image: require("./images/test2.jpeg"),
          logo: require('./images/logo.png')
        }
      }

    render(){
        return (
            <div style ={ { backgroundImage: "url("+this.state.image+")" } } className = 'header'>
                <div className = 'header-bar' >    
                    <div className="logo">
                        <img src={this.state.logo}/>
                        <p>ColorIT</p>
                    </div>
                    <div className="nav-menu">
                        <ul>
                            <li> <a className="selected" href='#'>Home</a></li>
                            <li><a href='#'>About Us</a></li>
                            <li><a href='#'>Try it</a></li>
                            <li><a href='#'>Our Team</a></li>
                        </ul>
                    </div>
                </div>

                <div className="main-text">
                    <h1>
                        Colorize your photos and videos
                    </h1>
                    <p>Using Deep Learning technique</p>
                </div>
            </div>
        )
    }
}

export default Header
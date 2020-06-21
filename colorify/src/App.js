import React from 'react';
import logo from './logo.svg';
import './App.css';
import Header from './components/header/header'
import Upload from './components/upload/upload'
import Team from './components/team/team'


function App() {
    return ( 
       <div className = "App" >
           <Header></Header>
           <Upload> </Upload>
           <Team></Team>
        </div>
    );
}

export default App;
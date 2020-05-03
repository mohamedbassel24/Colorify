import { Component, OnInit } from '@angular/core';
import { FormControl, Validators, FormGroup } from '@angular/forms';
import { MatDialog } from '@angular/material';
@Component({
  selector: 'app-contact-us',
  templateUrl: './contact-us.component.html',
  styleUrls: ['./contact-us.component.css']
})
export class ContactUsComponent implements OnInit {


  form: FormGroup;
  profileForm = new FormGroup({
   
    Name: new FormControl('', [Validators.required]),
    Email: new FormControl('', [Validators.required, Validators.email]),
    Phone: new FormControl('', [Validators.required]),
    Message: new FormControl('', [Validators.required]),
  });
  onSubmit() {
  }

  constructor() { }

  ngOnInit() {

    this.form = new FormGroup({
      // tslint:disable-next-line: max-line-length
      Name: new FormControl(null, { validators: [Validators.pattern('[^0-9\.\?\!\@\#\$\%\^\&\*\(\)\<\>\{\}]+')] }),
      Email: new FormControl(null, { validators: [Validators.required, Validators.email] }),
      Phone: new FormControl(null, { validators: [Validators.minLength(8)] }), // TODO: ADD validation on numbers 
      Message: new FormControl(null, { validators: [Validators.required] }),
 
    });


  }

}

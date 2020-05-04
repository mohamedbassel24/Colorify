import {Component, EventEmitter, OnInit, Output} from '@angular/core';
@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {
  @Output() private options: EventEmitter<File> = new EventEmitter();
  private invalidFiles: any = [];
  uploadError = false;
  messageError = '';
  private imageString: string | ArrayBuffer;
  private imageName: string;

  constructor() {
  }

  ngOnInit() {
  }

  onFilesChange(file: Array<File>) {
    if (this.uploadError === true) {
      return;
    }
    this.optionsMode(file[0]);
  }

  onFileInvalids(status: string) {
    if (status === 'invalid file') {
      this.uploadError = true;
      this.messageError = 'Invalid type image. The supported types are [jpg, jpeg, png]. Please, try another image';
    } else if (status === 'multiple files') {
      this.uploadError = true;
      this.messageError = 'You cannot upload multiple files. Just one image';
    }
  }

  optionsMode(imgFile: File) {
    const reader = new FileReader();
    reader.readAsDataURL(imgFile);
    reader.onload = () => {
      this.imageString = reader.result;
      this.imageName = imgFile.name;

    };
  }

  close() {
    this.uploadError = false;
    this.messageError = '';
  }
}

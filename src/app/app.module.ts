import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { MaterialModule } from './material.module';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { FooterComponent } from './footer/footer.component';
import { MainNavComponent } from './main-nav/main-nav.component';
import { ContactUsComponent } from './contact-us/contact-us.component';

import { MatFileUploadModule } from 'angular-material-fileupload';
import { MaterialFileInputModule } from 'ngx-material-file-input';
import { HomeComponent } from './home/home.component';
 
@NgModule({
  declarations: [
    AppComponent,
    FooterComponent,
    MainNavComponent,
    ContactUsComponent,
    HomeComponent,
    
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    MaterialModule,
    BrowserAnimationsModule,
    MatFileUploadModule,
    MaterialFileInputModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }

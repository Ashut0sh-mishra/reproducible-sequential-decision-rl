import { Component, OnInit } from '@angular/core';
import { ApiService } from '../../services/api.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.scss']
})
export class UploadComponent implements OnInit {

  selectedFile: File | null = null;
  message = '';
  progress = 0;

  uploadedInfo: any = null;
  uploadedFiles: any[] = [];   // ✅ SINGLE source

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.loadFiles();
  }

  onFileSelected(event: any) {
    this.selectedFile = event.target.files[0] ?? null;
  }

  upload() {
    if (!this.selectedFile) return;

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    this.progress = 0;
    this.message = 'Uploading...';

    this.api.uploadFile(formData).subscribe({
      next: (event: any) => {

        if (event.type === 1 && event.loaded && event.total) {
          this.progress = Math.round((event.loaded / event.total) * 100);
        }

        if (event.body) {
          this.uploadedInfo = event.body;
          this.message = 'Upload successful!';
          this.progress = 100;
          this.loadFiles(); // refresh list
        }
      },
      error: () => {
        this.message = 'Upload failed!';
      }
    });
  }

  loadFiles() {
    this.api.listFiles().subscribe({
      next: (res: any) => {
        this.uploadedFiles = res;
      },
      error: err => console.error(err)
    });
  }
}

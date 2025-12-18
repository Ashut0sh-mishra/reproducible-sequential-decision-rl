import { Component } from '@angular/core';
import { ApiService } from '../../services/api.service';
import { FormsModule } from '@angular/forms';
import { NgIf, JsonPipe } from '@angular/common';
import {  OnInit } from '@angular/core';

@Component({
  selector: 'app-experiment',
  standalone: true,
  imports: [FormsModule, NgIf, JsonPipe],
  templateUrl: './experiment.component.html',
  styleUrls: ['./experiment.component.scss']
})
export class ExperimentComponent implements OnInit{

  expId = "";
  results: any = null;
  loading = false;
  error = "";
  progress = 0;          // ✅ REQUIRED

  poller: any;

  constructor(private api: ApiService) {}

  loadResults() {
    if (!this.expId) {
      this.error = "Experiment ID cannot be empty.";
      return;
    }

    this.loading = true;
    this.error = "";
    this.results = null;
    this.progress = 0;

    this.poller = setInterval(() => {
      this.api.getExperimentResults(this.expId).subscribe({
        next: (res: any) => {
          this.results = res;

          if (res.progress?.progress !== undefined) {
            this.progress = res.progress.progress;
          }

         if (res.state === "SUCCESS" || res.state === "FAILURE") {
            clearInterval(this.poller);
            localStorage.removeItem('activeTaskId');
            this.loading = false;
          }
        },
        error: () => {
          this.error = "Backend error";
          clearInterval(this.poller);
          this.loading = false;
        }
      });
    }, 2000);
  }

  cancelTraining() {
  this.api.cancelTask(this.expId).subscribe(() => {
    clearInterval(this.poller);
    localStorage.removeItem('activeTaskId');
    this.loading = false;
    this.error = "Training cancelled";
    this.progress = 0;
  });
}

 ngOnInit() {
  const savedTask = localStorage.getItem('activeTaskId');

  if (savedTask) {
    this.expId = savedTask;
    this.loadResults(); // 🔄 auto-resume polling
  }
}


}

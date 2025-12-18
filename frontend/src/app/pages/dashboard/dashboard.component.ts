import { Component } from '@angular/core';
import { ApiService } from '../../services/api.service';
import { FormsModule } from '@angular/forms';
import { NgIf } from '@angular/common';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [FormsModule, NgIf],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent {
  degradationCost = 0.001;
  timesteps = 50000;

  taskId: string | null = null;

  constructor(private api: ApiService) {}

  startTraining() {
    const payload = {
      degradation_cost: this.degradationCost,
      timesteps: this.timesteps
    };

   this.api.startTraining(payload).subscribe({
  next: (res: any) => {
    this.taskId = res.task_id;

    // ✅ SAVE ACTIVE TASK ID
    localStorage.setItem('activeTaskId', res.task_id);
  },
  error: err => console.error(err)
});
  }
}

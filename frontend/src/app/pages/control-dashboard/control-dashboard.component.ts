import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-control-dashboard',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './control-dashboard.component.html',
  styleUrls: ['./control-dashboard.component.scss']
})
export class ControlDashboardComponent implements OnInit, OnDestroy {

  status: any = {
    fastapi: false,
    celery: false,
    flower: false,
    workers: []
  };

  private intervalId: any;

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.loadStatus();
    this.intervalId = setInterval(() => this.loadStatus(), 3000);
  }

  ngOnDestroy() {
    clearInterval(this.intervalId);
  }

  loadStatus() {
    this.api.getServiceStatus().subscribe({
      next: (res: any) => this.status = res,
      error: err => console.error(err)
    });
  }

  startCelery() {
    this.api.startCelery().subscribe(() => this.loadStatus());
  }

  stopCelery() {
    this.api.stopCelery().subscribe(() => this.loadStatus());
  }
}

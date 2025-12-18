import { Component, signal } from '@angular/core';

import { HttpClient } from '@angular/common/http';
import { environment } from '../environments/environment';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  protected readonly title = signal('frontend');

  constructor(private http: HttpClient) {}

  startServices() {
    this.http.post(environment.backendBaseUrl + 'control/start', {})
      .subscribe({
        next: () => alert('Starting all backend services...'),
        error: () => alert('Error: Backend is not reachable!')
      });
  }

  stopServices() {
    this.http.post(environment.backendBaseUrl + 'control/stop', {})
      .subscribe({
        next: () => alert('Stopping all backend services...'),
        error: () => alert('Error: Backend is not reachable!')
      });
  }
}

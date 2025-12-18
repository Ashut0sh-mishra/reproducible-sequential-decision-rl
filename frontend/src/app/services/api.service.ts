import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  private apiUrl = environment.backendBaseUrl; // http://localhost:9001

  constructor(private http: HttpClient) {}

  // ---------- UPLOAD ----------
  uploadFile(formData: FormData) {
    return this.http.post(`${this.apiUrl}/upload`, formData, {
      observe: 'events',
      reportProgress: true
    });
  }

  // ---------- TRAINING ----------
  startTraining(body: any) {
    return this.http.post(`${this.apiUrl}/experiment/start`, body);
  }

  getExperimentResults(id: string) {
    return this.http.get(`${this.apiUrl}/experiment/status/${id}`);
  }

  // ❌ CANCEL TRAINING (ADD THIS)
 cancelTask(taskId: string) {
  return this.http.post(
    `${this.apiUrl}/experiment/cancel/${taskId}`,
    {}
  );
}


  // ---------- FILES ----------
  listFiles() {
    return this.http.get(`${this.apiUrl}/upload/list`);
  }

  // ---------- CONTROL PANEL ----------
  getServiceStatus() {
    return this.http.get(`${this.apiUrl}/control/status`);
  }

  startCelery() {
    return this.http.post(`${this.apiUrl}/control/start`, {});
  }

  stopCelery() {
    return this.http.post(`${this.apiUrl}/control/stop`, {});
  }
}

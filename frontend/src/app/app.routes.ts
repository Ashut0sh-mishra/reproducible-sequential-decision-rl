import { Routes } from '@angular/router';
import { UploadComponent } from './pages/upload/upload.component';
import { DashboardComponent } from './pages/dashboard/dashboard.component';
import { ExperimentComponent } from './pages/experiment/experiment.component';
import { ControlDashboardComponent } from './pages/control-dashboard/control-dashboard.component';

export const routes: Routes = [
  { path: '', component: UploadComponent },
  { path: 'dashboard', component: DashboardComponent },
  { path: 'experiment', component: ExperimentComponent },
  { path: 'control', component: ControlDashboardComponent },
];

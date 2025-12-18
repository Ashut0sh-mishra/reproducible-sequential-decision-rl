import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ControlDashboardComponent } from './control-dashboard.component';

describe('ControlDashboardComponent', () => {
  let component: ControlDashboardComponent;
  let fixture: ComponentFixture<ControlDashboardComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ControlDashboardComponent]
    }).compileComponents();

    fixture = TestBed.createComponent(ControlDashboardComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

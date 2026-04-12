export type PresetKey = "xor" | "sine";

export interface TopologyState {
  structure: number[];
  learning_rate: number | null;
}

export interface LayerState {
  index: number;
  input_size: number;
  output_size: number;
  weights: number[][] | null;
  biases: number[] | null;
  last_input: number[] | null;
  last_pre_activation: number[] | null;
  last_output: number[] | null;
  last_gradient: number[] | null;
  learning_rate: number;
}

export interface CurrentSampleState {
  index: number;
  input: number[];
  target: number[];
  prediction: number[] | null;
  loss: number | null;
}

export interface TrainingState {
  current_epoch: number;
  current_sample_index: number;
  current_layer_index: number;
  phase: string;
  mode: string;
  epoch_losses: number[];
  total_samples: number;
  total_layers: number;
}

export interface SessionState {
  preset: PresetKey | null;
  topology: TopologyState;
  layers: LayerState[];
  network_output: number[] | null;
  training: TrainingState;
  current_sample: CurrentSampleState | null;
}

export interface SummaryState {
  training: TrainingState;
  network_output: number[] | null;
}

export interface PresetSummary {
  key: PresetKey;
  title: string;
  description: string;
  default_structure: number[];
  default_learning_rate: number;
  sample_count: number;
  input_size: number;
  output_size: number;
}

export interface ConfigureRequest {
  structure: number[];
  learning_rate: number;
  preset: PresetKey;
}

export interface StepEpochRequest {
  count: number;
}

export interface StepForwardRequest {
  sample_index?: number | null;
}

export interface RunRequest {
  speed_ms: number;
}

export type DashboardEvent =
  | { type: "state"; data: SessionState }
  | { type: "summary"; data: SummaryState }
  | { type: "error"; message: string };

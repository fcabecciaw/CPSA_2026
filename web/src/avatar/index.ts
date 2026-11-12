/**
 * The particle avatar of the dashboard (R2-F). Import from here: `AvatarView`
 * is the digital face, `VoiceEnergy` and the mood presets are the parts that can
 * be reused or unit-tested on their own.
 */

export { AvatarView, type AvatarDebug, type AvatarOptions } from "./view";
export { VoiceEnergy } from "./audio";
export {
  MOODS,
  MOOD_NAMES,
  MoodAnimator,
  isExpression,
  type MoodChannels,
  type MoodFrame,
  type MoodPose,
} from "./moods";
export { ParticleField, SPHERE_POINTS } from "./particles";

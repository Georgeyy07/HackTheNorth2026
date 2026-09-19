// Turn-by-turn progress tracking: given the backend's `directions` steps
// (each a turn point with lat/lon) and the phone's live GPS position, decides
// when to announce the next turn. Same haversine math as alert_math.py,
// reimplemented here since this runs on-device in JS, not through the API.

const EARTH_RADIUS_M = 6371000.0;

export function haversineDistanceM(lat1, lon1, lat2, lon2) {
  const toRad = (d) => (d * Math.PI) / 180;
  const phi1 = toRad(lat1);
  const phi2 = toRad(lat2);
  const dPhi = toRad(lat2 - lat1);
  const dLambda = toRad(lon2 - lon1);

  const a =
    Math.sin(dPhi / 2) ** 2 +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLambda / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return EARTH_RADIUS_M * c;
}

// How close (meters) counts as "arrived at this turn" -- announce it and move on.
export const TURN_ARRIVAL_RADIUS_M = 25;
// Give a heads-up this far out, once per step.
export const TURN_WARNING_RADIUS_M = 120;

/**
 * Tracks progress through a list of DirectionStep objects (from /api/route)
 * as GPS updates come in. Call `update(lat, lon)` on every position fix;
 * it returns an event describing what just happened, or null.
 */
export class TurnByTurnTracker {
  constructor(steps) {
    // Skip the "depart" step -- nothing to announce for it, we're already there.
    this.steps = steps.filter((s) => s.maneuver !== 'depart');
    this.currentIndex = 0;
    this.warnedForIndex = new Set();
  }

  get currentStep() {
    return this.steps[this.currentIndex] || null;
  }

  get isComplete() {
    return this.currentIndex >= this.steps.length;
  }

  update(lat, lon) {
    const step = this.currentStep;
    if (!step) return null;

    const distance = haversineDistanceM(lat, lon, step.lat, step.lon);

    if (distance <= TURN_ARRIVAL_RADIUS_M) {
      const completedStep = step;
      this.currentIndex += 1;
      return { type: completedStep.maneuver === 'arrive' ? 'arrived' : 'turn_now', step: completedStep, distance };
    }

    if (distance <= TURN_WARNING_RADIUS_M && !this.warnedForIndex.has(this.currentIndex)) {
      this.warnedForIndex.add(this.currentIndex);
      return { type: 'turn_ahead', step, distance };
    }

    return null;
  }
}

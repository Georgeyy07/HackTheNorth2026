/**
 * Fleet Video Synchronization Helper
 * Fully compatible with video_sync.json (sessions 2–5, variants staggered & cascade)
 */

export function resolveFleetVideo(sync, variant, carID, mapTimestampMs) {
  const scenarioKey = (variant || 'staggered').toLowerCase();

  // Extract car config from either video_sync.json schema (sync.variants[scenarioKey].cars)
  // or direct dictionary format
  let carCfg = null;
  if (sync && sync.variants && sync.variants[scenarioKey] && Array.isArray(sync.variants[scenarioKey].cars)) {
    carCfg = sync.variants[scenarioKey].cars.find(c => {
      if (!c.carID) return false;
      if (c.carID === carID) return true;
      for (const num of ['01', '02', '03', '04']) {
        if (c.carID.endsWith(num) && (carID.endsWith(num) || carID.includes(`-${num}`))) {
          return true;
        }
      }
      return false;
    });
  } else if (sync && sync[scenarioKey]) {
    carCfg = sync[scenarioKey][carID] || Object.values(sync[scenarioKey]).find(c => c.carID === carID);
  }

  // Fallback defaults if car not found in sync manifest
  if (!carCfg) {
    let firstSess = 'session2';
    let launchOffset = 8.902;
    let delay = 0;
    if (carID.includes('02')) {
      firstSess = scenarioKey === 'cascade' ? 'session2' : 'session3';
      launchOffset = scenarioKey === 'cascade' ? 8.902 : 0.118;
      delay = scenarioKey === 'cascade' ? 20 : 0;
    } else if (carID.includes('03')) {
      firstSess = scenarioKey === 'cascade' ? 'session2' : 'session4';
      launchOffset = scenarioKey === 'cascade' ? 8.902 : 0.043;
      delay = scenarioKey === 'cascade' ? 40 : 0;
    } else if (carID.includes('04')) {
      firstSess = scenarioKey === 'cascade' ? 'session2' : 'session5';
      launchOffset = scenarioKey === 'cascade' ? 8.902 : 0.294;
      delay = scenarioKey === 'cascade' ? 60 : 0;
    }
    carCfg = {
      firstSession: firstSess,
      videoAtLaunchSeconds: launchOffset,
      launchDelaySeconds: delay,
    };
  }

  const baseStartMs = Date.parse(sync?.start || '2026-09-20T12:00:00Z');
  const launchDelaySeconds = carCfg.launch_delay_s !== undefined 
    ? carCfg.launch_delay_s 
    : (carCfg.launchDelaySeconds || 0);
  const launchTimestampMs = baseStartMs + (launchDelaySeconds * 1000);

  const firstSession = carCfg.source_start?.session || carCfg.firstSession || 'session2';

  if (mapTimestampMs < launchTimestampMs) {
    const waitSec = ((launchTimestampMs - mapTimestampMs) / 1000).toFixed(1);
    return {
      state: 'waiting',
      session: firstSession,
      url: `/demo_view/videos/${firstSession}.mp4`,
      currentTime: 0,
      message: `Launches in ${waitSec}s (${firstSession})`
    };
  }

  // Exact durations from video_sync.json
  const sessionOrder = ['session2', 'session3', 'session4', 'session5'];
  const sessionDurations = {
    session2: 261.91,
    session3: 231.11,
    session4: 675.09,
    session5: 138.77,
  };
  const gaps = {
    session2: 1.610,
    session3: 5.615,
    session4: 3.919,
  };
  const defaultLaunchOffsets = {
    session2: 8.902,
    session3: 0.118,
    session4: 0.043,
    session5: 0.294,
  };

  const startIndex = sessionOrder.indexOf(firstSession) >= 0 ? sessionOrder.indexOf(firstSession) : 0;
  let initialOffset = defaultLaunchOffsets[firstSession] || 0;
  if (carCfg.videoAtLaunchSeconds !== undefined) {
    initialOffset = carCfg.videoAtLaunchSeconds;
  } else if (carCfg.source_start && carCfg.source_start.source_start_s !== undefined) {
    const vOffset = { session2: 0.378, session3: 0.438, session4: 0.363, session5: 0.614 }[firstSession] || 0;
    initialOffset = Math.max(0, carCfg.source_start.source_start_s - vOffset);
  }

  let elapsedSinceLaunchSec = (mapTimestampMs - launchTimestampMs) / 1000.0;

  for (let i = startIndex; i < sessionOrder.length; i++) {
    const sessName = sessionOrder[i];
    const dur = sessionDurations[sessName];
    const offset = (i === startIndex) ? initialOffset : 0;
    const remainingInSession = dur - offset;

    if (elapsedSinceLaunchSec <= remainingInSession) {
      const currentVideoTime = offset + elapsedSinceLaunchSec;
      return {
        state: 'playing',
        session: sessName,
        url: `/demo_view/videos/${sessName}.mp4`,
        currentTime: Math.max(0, currentVideoTime),
        duration: dur,
        message: `${sessName.toUpperCase()} @ ${currentVideoTime.toFixed(1)}s`
      };
    }

    elapsedSinceLaunchSec -= remainingInSession;

    const gapSec = gaps[sessName] || 0;
    if (elapsedSinceLaunchSec < gapSec) {
      return {
        state: 'recording_gap',
        session: sessName,
        url: `/demo_view/videos/${sessName}.mp4`,
        currentTime: dur,
        message: `Recording Gap (${gapSec.toFixed(1)}s between ${sessName} and ${sessionOrder[i+1]})`
      };
    }

    elapsedSinceLaunchSec -= gapSec;
  }

  return {
    state: 'finished',
    session: 'session5',
    url: '/demo_view/videos/session5.mp4',
    currentTime: sessionDurations['session5'],
    message: 'Fleet Patrol Mission Completed'
  };
}

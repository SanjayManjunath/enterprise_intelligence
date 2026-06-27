type RateLimitRecord = {
  count: number;
  resetTime: number;
};

// Simple, lightweight in-memory storage maps
const ipCache = new Map<string, RateLimitRecord>();
let globalTokenCount = 0;
let globalTokenWindowReset = Date.now() + 60000; // 1 minute window

export interface RateLimitResponse {
  allowed: boolean;
  error?: string;
  statusCode?: number;
}

export function checkRateLimits(ip: string, estimatedTokens: number): RateLimitResponse {
  const now = Date.now();

  // 1. --- GLOBAL TOKEN SHIELD ---
  // Reset window if 60 seconds have elapsed
  if (now > globalTokenWindowReset) {
    globalTokenCount = 0;
    globalTokenWindowReset = now + 60000;
  }

  // Hard fuse: check if adding these tokens blows past the 25k limit
  if (globalTokenCount + estimatedTokens > 25000) {
    return {
      allowed: false,
      error: "Cloud Compute Cooldown Active: Global API throughput threshold reached. Retry in 1 minute.",
      statusCode: 429
    };
  }

  // 2. --- PER-IP RATE LIMITER (10 requests per 15 mins) ---
  const windowMs = 15 * 60 * 1000; 
  const clientRecord = ipCache.get(ip);

  if (!clientRecord || now > clientRecord.resetTime) {
    // Fresh record or expired window
    ipCache.set(ip, {
      count: 1,
      resetTime: now + windowMs
    });
  } else {
    // Within existing window
    if (clientRecord.count >= 10) {
      const remainingSeconds = Math.ceil((clientRecord.resetTime - now) / 1000);
      const remainingMinutes = Math.ceil(remainingSeconds / 60);
      return {
        allowed: false,
        error: `Rate Limit Exceeded: You can perform 10 database audits every 15 minutes. Please wait ${remainingMinutes} minute(s).`,
        statusCode: 429
      };
    }
    clientRecord.count += 1;
  }

  // Commit the tokens to the global fuse if both checks pass
  globalTokenCount += estimatedTokens;
  return { allowed: true };
}
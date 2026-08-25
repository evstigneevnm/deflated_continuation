#ifndef __STABILITY_DETAIL_LEGACY_API_H__
#define __STABILITY_DETAIL_LEGACY_API_H__

#if defined(STABILITY_SUPPRESS_LEGACY_DEPRECATION_WARNINGS)
#define STABILITY_LEGACY_API(message)
#else
#define STABILITY_LEGACY_API(message) [[deprecated(message)]]
#endif

#endif

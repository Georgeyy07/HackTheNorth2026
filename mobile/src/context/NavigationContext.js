import React, { createContext, useContext, useState } from 'react';

const NavigationContext = createContext({
  isNavigating: false,
  setIsNavigating: () => {},
});

export function NavigationProvider({ children }) {
  const [isNavigating, setIsNavigating] = useState(false);

  return (
    <NavigationContext.Provider value={{ isNavigating, setIsNavigating }}>
      {children}
    </NavigationContext.Provider>
  );
}

export function useNavigationStatus() {
  return useContext(NavigationContext);
}

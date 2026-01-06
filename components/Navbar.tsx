
import React from 'react';

const Navbar: React.FC = () => {
  return (
    <header className="sticky top-0 z-50 flex items-center justify-between border-b border-solid border-border-dark bg-background-dark/80 backdrop-blur-md px-6 py-4 lg:px-10">
      <div className="flex items-center gap-4 text-white">
        <div className="size-8 flex items-center justify-center rounded-lg bg-gradient-to-br from-primary to-blue-600 text-white shadow-lg shadow-primary/20">
          <span className="material-symbols-outlined text-xl">lightbulb</span>
        </div>
        <h2 className="text-white text-lg font-bold leading-tight tracking-tight">Startup Idea Novelty Visualizer</h2>
      </div>
      <div className="flex gap-3">
        <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
          <span className="material-symbols-outlined text-[16px]">verified_user</span>
          <span>Client-Side Secure</span>
        </div>
        <button className="flex items-center justify-center rounded-lg h-9 px-4 bg-surface-dark hover:bg-border-dark text-white text-sm font-medium transition-colors border border-border-dark">
          About
        </button>
      </div>
    </header>
  );
};

export default Navbar;

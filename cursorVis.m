function cursorVis(state, goal, actualUpdate, oracleUpdate, bounds, figNum)

    % the update inputs are relative vectors (like velocity)
    
    figure(figNum)

    % plot goal
    plot3(state(1),state(2),state(3),'.','markerSize',40)

    % plot state
    hold on
    plot3(goal(1),goal(2),goal(3),'g.','markerSize',40)
    hold off
    
    % change size of plot
    xlim(bounds{1})
    ylim(bounds{1})
    zlim(bounds{1})
    
    % plot actual update
    hold on
    quiver3(state(1),state(2),state(3),actualUpdate(1),actualUpdate(2),actualUpdate(3),'r','LineWidth',2)
    
    % plot oracle update
    quiver3(state(1),state(2),state(3),oracleUpdate(1),oracleUpdate(2),oracleUpdate(3),'--g','LineWidth',2)
    hold off
    
    grid on
    
end

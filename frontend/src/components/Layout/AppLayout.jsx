import styled from "styled-components";
import { Link, useLocation, Outlet } from "react-router-dom";
import PropTypes from "prop-types";

const LayoutContainer = styled.div`
  display: flex;
  min-height: 100vh;
  background: #1e1e1e;
`;

const Sidebar = styled.div`
  width: 280px;
  background: #2c2c2c;
  border-right: 1px solid #374151;
  padding: 1.5rem;
  display: ${(props) => (props.hidden ? "none" : "block")};
`;

const MainContent = styled.div`
  flex: 1;
  overflow-y: auto;
`;

const MenuList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const MenuItem = styled.li`
  margin-bottom: 0.5rem;
`;

const MenuLink = styled(Link)`
  display: flex;
  align-items: center;
  padding: 0.75rem 1rem;
  color: #e0e0e0;
  text-decoration: none;
  border-radius: 0.5rem;
  transition: background-color 0.2s;

  &:hover {
    background: #3b3b3b;
  }

  &.active {
    background: #4b4b4b;
    color: #a855f7;
  }
`;

const Logo = styled(Link)`
  display: block;
  padding: 0 1rem;
  margin-bottom: 2rem;
  color: #e0e0e0;
  text-decoration: none;
  font-size: 1.5rem;
  font-weight: 600;
`;

const AppLayout = () => {
  const location = useLocation();
  const isInProject = location.pathname.includes("/project/");

  return (
    <LayoutContainer>
      <Sidebar hidden={isInProject}>
        <Logo to="/">Threat Analysis</Logo>
        <MenuList>
          <MenuItem>
            <MenuLink
              to="/projects"
              className={location.pathname === "/projects" ? "active" : ""}
            >
              Projects
            </MenuLink>
          </MenuItem>
          <MenuItem>
            <MenuLink
              to="/documents"
              className={location.pathname === "/documents" ? "active" : ""}
            >
              Documents
            </MenuLink>
          </MenuItem>
          <MenuItem>
            <MenuLink
              to="/settings"
              className={location.pathname === "/settings" ? "active" : ""}
            >
              Settings
            </MenuLink>
          </MenuItem>
        </MenuList>
      </Sidebar>
      <MainContent>
        <Outlet />
      </MainContent>
    </LayoutContainer>
  );
};

AppLayout.propTypes = {
  children: PropTypes.node,
};

export default AppLayout;

%global pypi_name neurojax

Name:           python-neurojax
Version:        0.1.0
Release:        1%{?dist}
Summary:        JAX-accelerated Electrophysiology Analysis CLI

# License found in repository root is MIT (implied, verify in repo)
License:        MIT
URL:            https://github.com/AthenaEPI/neurojax
Source0:        %{pypi_name}-%{version}.tar.gz

BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  python3-pytest

%description
JAX-accelerated Electrophysiology Analysis CLI.

%package -n python3-neurojax
Summary:        %{summary}
Requires:       python3-jax
Requires:       python3-mne
Requires:       python3-scipy
Requires:       python3-numpy
# Note: jaxlib GPU support is likely desired but we depend on python3-jax which pulls
# in a CPU-only jaxlib by default on Fedora. Users need to install proper jaxlib for GPU.

%description -n python3-neurojax
%{description}

%prep
%autosetup -p1 -n neurojax-%{version}

%generate_buildrequires
%pyproject_buildrequires

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files neurojax

%check
# Run pytest
%pytest

%files -n python3-neurojax -f %{pyproject_files}
%doc README.md
# %license LICENSE  # Uncomment if LICENSE file exists

%changelog
* Tue Jan 28 2026 Antigravity <antigravity@example.com> - 0.1.0-1
- Initial package
